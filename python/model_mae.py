"""
model_mae.py
============
Masked AutoEncoder (MAE) 기반 회전체 진동 이상 탐지 모델.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
왜 MAE인가?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기존 CNN AE 한계:
  - 단순 AE는 국소 패턴(컨볼루션)에 집중 → 장주기 회전 패턴 포착 약함
  - 입력을 그대로 복사하는 trivial identity mapping 학습 위험

MAE 해결책:
  - 전체 신호의 75%를 마스킹 후 가시 패치만으로 마스킹 패치 재구성
  - 고마스킹비 → 모델이 신호의 전체적 구조(주기성, 기저 패턴)를 이해해야만 재구성 가능
  - Transformer → 패치 간 장거리 의존성 학습 (전체 orbit 형태, 축 간 위상 관계)
  - 정상 데이터만으로 학습 → 비정상 패턴 재구성 실패 = 이상 점수

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
두 스트림 구성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OrbitMAE1D  — 시간 도메인 (X, Y 2채널 원시 신호)
  입력 : (B, 2, 40000) 고정 스케일
  패치 : 500 샘플 × 2채널 = 1000-dim, N=80 패치
  마스킹: 75% → 20 가시 패치로 60 패치 재구성
  장점 : 절대 진폭 + 파형 형태 동시 인코딩

OrbitMAESpec — 주파수 도메인 (4채널 로그 스펙트로그램)
  입력 : (B, 4, 257, 157) 고정 스케일 → 패딩 후 (B, 4, 272, 160)
  패치 : 16×8 픽셀 × 4채널 = 512-dim, N=340 패치
  마스킹: 85% → 51 가시 패치로 289 패치 재구성
  장점 : 주파수 에너지 분포 (0–10 kHz) + 교차 스펙트럼 위상차 (와류 방향)

OrbitMAE  — 통합 래퍼
  두 스트림 결합 학습, 통합 이상 점수 반환

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이상 점수 (Anomaly Score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
여러 번 랜덤 마스크 적용 후 평균 재구성 오차 (Monte Carlo):
  score = (1/K) Σ_k MSE(pred[masked_k], target[masked_k])

고정 스케일 입력이므로 MSE에 절대 진폭 편차가 반영됨:
  정상 신호: 낮은 재구성 오차 (익숙한 패턴)
  비정상 신호: 높은 재구성 오차 (미학습 패턴 + 진폭 이상)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 상수 (model-level defaults)
# ─────────────────────────────────────────────

# 1D 브랜치
SEQ_LEN        = 40_000
IN_CH_1D       = 2         # X, Y
PATCH_SIZE_1D  = 500       # 500 샘플/패치 → N=80 패치
PATCH_DIM_1D   = IN_CH_1D * PATCH_SIZE_1D   # 1000

# 스펙트로그램 브랜치
IN_CH_SPEC     = 4         # Sx, Sy, Re(Gxy), Im(Gxy)
SPEC_F_BINS    = 257       # 주파수 빈 (0–10 kHz at 39.1 Hz/bin) — 10000/39.0625 = 256 + 1
SPEC_T_FRAMES  = 157       # 시간 프레임 (1초 신호, hop=256)
SPEC_PATCH_H   = 16        # 주파수 패치 크기 (16 빈 × 39.1 Hz = ~625 Hz/패치)
SPEC_PATCH_W   = 8         # 시간 패치 크기

# Transformer 아키텍처
D_ENC         = 256
D_DEC         = 128
N_ENC_LAYERS  = 4
N_DEC_LAYERS  = 2
N_HEADS_ENC   = 8
N_HEADS_DEC   = 4
D_FF_ENC      = 1024
D_FF_DEC      = 512
MASK_RATIO    = 0.75
DROPOUT       = 0.1


# ─────────────────────────────────────────────
# 위치 인코딩
# ─────────────────────────────────────────────

def _sincos_1d(seq_len: int, d: int) -> torch.Tensor:
    """
    1D 사인/코사인 위치 인코딩 (학습되지 않는 고정 인코딩).
    Returns: (seq_len, d)
    """
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)   # (N, 1)
    div = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d)
    )  # (d//2,)
    pe = torch.zeros(seq_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d // 2])  # 홀수 d 처리
    return pe


def _sincos_2d(n_h: int, n_w: int, d: int) -> torch.Tensor:
    """
    2D 사인/코사인 위치 인코딩.
    높이 축 d//2 + 너비 축 d//2 를 concat.
    Returns: (n_h*n_w, d)
    """
    assert d % 4 == 0, f"2D sincos requires d % 4 == 0, got d={d}"
    half = d // 2
    pe_h = _sincos_1d(n_h, half)                           # (n_h, d//2)
    pe_w = _sincos_1d(n_w, half)                           # (n_w, d//2)
    pe_h = pe_h.unsqueeze(1).expand(-1, n_w, -1)           # (n_h, n_w, d//2)
    pe_w = pe_w.unsqueeze(0).expand(n_h, -1, -1)           # (n_h, n_w, d//2)
    return torch.cat([pe_h, pe_w], dim=-1).reshape(n_h * n_w, d)


# ─────────────────────────────────────────────
# Transformer Block (Pre-LN)
# ─────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
      x → LN → MHA → residual → LN → FFN → residual
    Pre-LN은 Post-LN보다 학습 안정성이 높아 적은 데이터에서 유리함.
    """

    def __init__(self, d: int, nhead: int, d_ff: int, dropout: float = DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(
            d, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        h = self.norm2(x)
        h = self.ff(h)
        return x + self.drop(h)


# ─────────────────────────────────────────────
# 마스킹 유틸
# ─────────────────────────────────────────────

def _random_mask(
    B: int, N: int, mask_ratio: float, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    배치별 독립 랜덤 마스킹.

    Returns:
        ids_keep    : (B, n_vis)  가시 패치 인덱스
        ids_restore : (B, N)     원래 순서 복원 인덱스 (역순열)
    """
    n_vis = max(1, int(N * (1.0 - mask_ratio)))
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)           # 오름차순 → 앞쪽 n_vis가 가시
    ids_restore  = torch.argsort(ids_shuffle, dim=1)   # 역순열
    return ids_shuffle[:, :n_vis], ids_restore


def _make_bool_mask(ids_restore: torch.Tensor, n_vis: int) -> torch.Tensor:
    """
    Bool 마스크 생성: True = 마스킹된(재구성 대상) 위치.
    Returns: (B, N)
    """
    B, N = ids_restore.shape
    mask = torch.ones(B, N, device=ids_restore.device)
    mask[:, :n_vis] = 0.0
    # 셔플 역방향 적용 (원래 패치 순서로 복원)
    return torch.gather(mask, 1, ids_restore).bool()


# ─────────────────────────────────────────────
# 1D MAE
# ─────────────────────────────────────────────

class OrbitMAE1D(nn.Module):
    """
    시간 도메인 1D Masked AutoEncoder.

    입력 : (B, 2, 40000)  ← prepare_1d_input_fixed() 출력
    학습 : MSE on masked patches (정상 데이터만)
    추론 : 여러 랜덤 마스크에 대한 평균 재구성 오차 → 이상 점수

    아키텍처:
      Patchify (500샘플/패치, N=80)
      → Patch Embed (1000 → 256)
      → 랜덤 마스킹 75% (20패치만 인코더 통과)
      → Encoder: 4× TransformerBlock(256, 8head, ff=1024)
      → Decoder: enc_to_dec + mask_token 삽입 + 2× TransformerBlock(128, 4head, ff=512)
      → Recon Head (128 → 1000)
      → Loss: MSE on 60 masked patches
    """

    def __init__(
        self,
        in_channels:  int   = IN_CH_1D,
        seq_len:      int   = SEQ_LEN,
        patch_size:   int   = PATCH_SIZE_1D,
        d_enc:        int   = D_ENC,
        d_dec:        int   = D_DEC,
        n_enc_layers: int   = N_ENC_LAYERS,
        n_dec_layers: int   = N_DEC_LAYERS,
        n_heads_enc:  int   = N_HEADS_ENC,
        n_heads_dec:  int   = N_HEADS_DEC,
        d_ff_enc:     int   = D_FF_ENC,
        d_ff_dec:     int   = D_FF_DEC,
        mask_ratio:   float = MASK_RATIO,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len({seq_len}) must be divisible by patch_size({patch_size})"

        self.in_channels = in_channels
        self.seq_len     = seq_len
        self.patch_size  = patch_size
        self.n_patches   = seq_len // patch_size       # 80
        self.patch_dim   = in_channels * patch_size    # 1000
        self.d_enc       = d_enc
        self.d_dec       = d_dec
        self.mask_ratio  = mask_ratio
        self.n_vis       = max(1, int(self.n_patches * (1.0 - mask_ratio)))  # 20

        # ── Encoder ──────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_dim, d_enc, bias=True)
        self.register_buffer(
            "enc_pos",
            _sincos_1d(self.n_patches, d_enc).unsqueeze(0),   # (1, N, d_enc)
        )
        self.encoder  = nn.ModuleList(
            [_TransformerBlock(d_enc, n_heads_enc, d_ff_enc, dropout)
             for _ in range(n_enc_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_enc)

        # ── Decoder ──────────────────────────────────
        self.enc_to_dec = nn.Linear(d_enc, d_dec, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        self.register_buffer(
            "dec_pos",
            _sincos_1d(self.n_patches, d_dec).unsqueeze(0),   # (1, N, d_dec)
        )
        self.decoder    = nn.ModuleList(
            [_TransformerBlock(d_dec, n_heads_dec, d_ff_dec, dropout)
             for _ in range(n_dec_layers)]
        )
        self.dec_norm   = nn.LayerNorm(d_dec)
        self.recon_head = nn.Linear(d_dec, self.patch_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Patch 변환 ─────────────────────────────────

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, N, C×P)"""
        B, C, L = x.shape
        x = x.reshape(B, C, self.n_patches, self.patch_size)   # (B, C, N, P)
        x = x.permute(0, 2, 1, 3)                               # (B, N, C, P)
        return x.reshape(B, self.n_patches, self.patch_dim)     # (B, N, C*P)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C×P) → (B, C, L)"""
        B, N, _ = x.shape
        x = x.reshape(B, N, self.in_channels, self.patch_size)  # (B, N, C, P)
        x = x.permute(0, 2, 1, 3)                               # (B, C, N, P)
        return x.reshape(B, self.in_channels, self.seq_len)     # (B, C, L)

    # ── Forward ─────────────────────────────────────

    def forward_masked(
        self,
        x:          torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE 순전파 (마스킹 포함).

        Args:
            x          : (B, C, L) 입력 신호
            mask_ratio : 마스킹 비율 (None이면 기본값 사용)

        Returns:
            loss            : 스칼라, 마스킹 패치에 대한 평균 MSE (역전파용)
            per_sample_loss : (B,) 샘플별 마스킹 패치 MSE (이상 점수용)
            mask            : (B, N) bool, True = 마스킹된 위치
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B = x.size(0)
        target = self.patchify(x)   # (B, N, patch_dim)

        # 1) Patch embedding + positional encoding
        tokens = self.patch_embed(target) + self.enc_pos    # (B, N, d_enc)

        # 2) Random masking
        ids_keep, ids_restore = _random_mask(B, self.n_patches, mask_ratio, x.device)
        n_vis = ids_keep.size(1)

        # 3) 가시 토큰만 선택
        vis = torch.gather(
            tokens, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.d_enc),
        )   # (B, n_vis, d_enc)

        # 4) Encoder
        for blk in self.encoder:
            vis = blk(vis)
        vis = self.enc_norm(vis)    # (B, n_vis, d_enc)

        # 5) Decoder
        pred = self._decode(vis, ids_restore, n_vis)        # (B, N, patch_dim)

        # 6) 마스크 생성
        mask = _make_bool_mask(ids_restore, n_vis)          # (B, N) bool

        # 7) 손실: 마스킹 패치에 대한 MSE
        err        = (pred - target).pow(2).mean(dim=-1)    # (B, N) patch-wise MSE
        masked_err = (err * mask.float()).sum(dim=1) \
                   / mask.float().sum(dim=1).clamp(min=1)   # (B,) per-sample

        return masked_err.mean(), masked_err, mask

    def _decode(
        self,
        vis:         torch.Tensor,
        ids_restore: torch.Tensor,
        n_vis:       int,
    ) -> torch.Tensor:
        """인코더 출력 → 전체 N 패치 재구성."""
        B = vis.size(0)
        N = ids_restore.size(1)
        n_mask = N - n_vis

        # enc → dec 차원 변환
        vis_dec = self.enc_to_dec(vis)                          # (B, n_vis, d_dec)

        # 마스크 토큰 추가
        mask_tokens = self.mask_token.expand(B, n_mask, -1)    # (B, n_mask, d_dec)
        x = torch.cat([vis_dec, mask_tokens], dim=1)           # (B, N, d_dec)

        # 원래 패치 순서 복원 (unshuffle)
        x = torch.gather(
            x, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.d_dec),
        )   # (B, N, d_dec)

        # 디코더 위치 인코딩 추가
        x = x + self.dec_pos                                   # (B, N, d_dec)

        # Decoder Transformer
        for blk in self.decoder:
            x = blk(x)
        x = self.dec_norm(x)                                   # (B, N, d_dec)

        return self.recon_head(x)                              # (B, N, patch_dim)

    # ── 이상 점수 ───────────────────────────────────

    @torch.no_grad()
    def _score_per_patch(
        self,
        x:          torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        패치별 MSE (B, N) + bool 마스크 (B, N) 반환.
        top-k 집계용 내부 메서드 — forward_masked 인터페이스 불변 유지.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B      = x.size(0)
        target = self.patchify(x)
        tokens = self.patch_embed(target) + self.enc_pos
        ids_keep, ids_restore = _random_mask(B, self.n_patches, mask_ratio, x.device)
        n_vis  = ids_keep.size(1)
        vis    = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.d_enc))
        for blk in self.encoder:
            vis = blk(vis)
        vis  = self.enc_norm(vis)
        pred = self._decode(vis, ids_restore, n_vis)
        mask = _make_bool_mask(ids_restore, n_vis)
        err  = (pred - target).pow(2).mean(dim=-1)  # (B, N)
        return err, mask

    def anomaly_score(
        self,
        x:          torch.Tensor,
        n_eval:     int          = 10,
        mask_ratio: float | None = None,
        topk_ratio: float        = 1.0,
    ) -> torch.Tensor:
        """
        Monte Carlo 마스킹으로 안정적인 이상 점수 계산.

        Args:
            x          : (B, C, L)
            n_eval     : 랜덤 마스크 반복 횟수 (높을수록 안정, 느려짐)
            mask_ratio : 마스킹 비율 (기본값 사용 권장)
            topk_ratio : 오차 상위 K% 패치 평균 사용 (1.0=전체 평균, 0.1=상위 10%)
                         1.0 미만 시 transient 등 국소 이상 탐지 강화

        Returns:
            score : (B,) 이상 점수, 높을수록 이상
        """
        scores = torch.zeros(x.size(0), device=x.device)
        if topk_ratio >= 1.0:
            for _ in range(n_eval):
                _, per_sample, _ = self.forward_masked(x, mask_ratio)
                scores += per_sample
        else:
            # Top-k 집계: 오차 상위 K% 마스킹 패치 평균
            for _ in range(n_eval):
                err, mask = self._score_per_patch(x, mask_ratio)  # (B,N), (B,N)
                masked_err = err * mask.float()                    # 가시 패치 → 0
                n_masked = int(mask[0].sum().item())
                k_num    = max(1, int(n_masked * topk_ratio))
                topk_vals, _ = masked_err.topk(k_num, dim=1, largest=True)  # (B, k)
                scores += topk_vals.mean(dim=1)
        return scores / n_eval

    @torch.no_grad()
    def reconstruct_once(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        단일 마스크 포워드 — 재구성 신호 + 패치별 오차 반환 (시각화용).

        Returns:
            recon   : (B, C, L)  재구성된 신호
            err_map : (B, N)     패치별 MSE (마스킹 위치만, 나머지 0)
            mask    : (B, N)     bool, True = 마스킹 위치
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B = x.size(0)
        target = self.patchify(x)
        tokens = self.patch_embed(target) + self.enc_pos
        ids_keep, ids_restore = _random_mask(B, self.n_patches, mask_ratio, x.device)
        n_vis = ids_keep.size(1)
        vis = torch.gather(
            tokens, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.d_enc),
        )
        for blk in self.encoder:
            vis = blk(vis)
        vis = self.enc_norm(vis)
        pred    = self._decode(vis, ids_restore, n_vis)
        mask    = _make_bool_mask(ids_restore, n_vis)
        err_map = (pred - target).pow(2).mean(-1) * mask.float()
        recon   = self.unpatchify(pred)
        return recon, err_map, mask

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        마스킹 없이 전체 패치 인코딩 → 평균 풀링된 표현 벡터.
        SVDD 3단계 등 다운스트림에 활용 가능.
        Returns: (B, d_enc)
        """
        target = self.patchify(x)
        tokens = self.patch_embed(target) + self.enc_pos
        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.enc_norm(tokens)  # (B, N, d_enc)
        return tokens.mean(dim=1)       # (B, d_enc)

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        """학습 편의를 위한 forward_masked 별칭."""
        return self.forward_masked(x, mask_ratio)


# ─────────────────────────────────────────────
# 스펙트로그램 MAE
# ─────────────────────────────────────────────

class OrbitMAESpec(nn.Module):
    """
    주파수 도메인 2D Masked AutoEncoder.

    입력 : (B, 4, 257, 157)  ← make_spectrogram_4ch() 출력
           (F, T가 다소 다를 경우 자동 패딩/크롭)
    패딩 후 (B, 4, 272, 160) → 17×20=340 패치
    패치 크기: 16×8 × 4ch = 512 dim

    Ch0: Sx (X 전력)  Ch1: Sy (Y 전력)
    Ch2: Re(Gxy)      Ch3: Im(Gxy) ← 와류 방향 인코딩
    """

    def __init__(
        self,
        in_channels:  int   = IN_CH_SPEC,
        f_bins:       int   = SPEC_F_BINS,
        t_frames:     int   = SPEC_T_FRAMES,
        patch_h:      int   = SPEC_PATCH_H,
        patch_w:      int   = SPEC_PATCH_W,
        d_enc:        int   = D_ENC,
        d_dec:        int   = D_DEC,
        n_enc_layers: int   = N_ENC_LAYERS,
        n_dec_layers: int   = N_DEC_LAYERS,
        n_heads_enc:  int   = N_HEADS_ENC,
        n_heads_dec:  int   = N_HEADS_DEC,
        d_ff_enc:     int   = D_FF_ENC,
        d_ff_dec:     int   = D_FF_DEC,
        mask_ratio:   float = MASK_RATIO,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        assert d_enc % 4 == 0, "d_enc must be divisible by 4 for 2D sincos encoding"
        assert d_dec % 4 == 0, "d_dec must be divisible by 4 for 2D sincos encoding"

        self.in_channels = in_channels
        self.patch_h     = patch_h
        self.patch_w     = patch_w
        self.d_enc       = d_enc
        self.d_dec       = d_dec
        self.mask_ratio  = mask_ratio

        # 패딩된 크기 (패치 크기의 배수)
        self.pad_f = math.ceil(f_bins   / patch_h) * patch_h  # 272
        self.pad_t = math.ceil(t_frames / patch_w) * patch_w  # 160
        self.n_h   = self.pad_f // patch_h                    # 17
        self.n_w   = self.pad_t // patch_w                    # 20

        self.n_patches = self.n_h * self.n_w                  # 140
        self.patch_dim = in_channels * patch_h * patch_w      # 256
        self.n_vis     = max(1, int(self.n_patches * (1.0 - mask_ratio)))  # 35

        # ── Encoder ──────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_dim, d_enc, bias=True)
        self.register_buffer(
            "enc_pos",
            _sincos_2d(self.n_h, self.n_w, d_enc).unsqueeze(0),  # (1, N, d_enc)
        )
        self.encoder  = nn.ModuleList(
            [_TransformerBlock(d_enc, n_heads_enc, d_ff_enc, dropout)
             for _ in range(n_enc_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_enc)

        # ── Decoder ──────────────────────────────────
        self.enc_to_dec = nn.Linear(d_enc, d_dec, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        self.register_buffer(
            "dec_pos",
            _sincos_2d(self.n_h, self.n_w, d_dec).unsqueeze(0),  # (1, N, d_dec)
        )
        self.decoder    = nn.ModuleList(
            [_TransformerBlock(d_dec, n_heads_dec, d_ff_dec, dropout)
             for _ in range(n_dec_layers)]
        )
        self.dec_norm   = nn.LayerNorm(d_dec)
        self.recon_head = nn.Linear(d_dec, self.patch_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Patch 변환 ─────────────────────────────────

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, F, T) → (B, N, C×pH×pW)
        F, T가 pad_f, pad_t와 다를 경우 자동 패딩/크롭.
        """
        B, C, Fq, Tq = x.shape

        # 패딩 (부족한 경우)
        pad_f = max(0, self.pad_f - Fq)
        pad_t = max(0, self.pad_t - Tq)
        if pad_f > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_f))

        # 크롭 (초과하는 경우)
        x = x[:, :, :self.pad_f, :self.pad_t]

        # (B, C, n_h, pH, n_w, pW) → (B, N, C*pH*pW)
        x = x.reshape(B, C, self.n_h, self.patch_h, self.n_w, self.patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5)              # (B, n_h, n_w, C, pH, pW)
        return x.reshape(B, self.n_patches, self.patch_dim)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C×pH×pW) → (B, C, pad_f, pad_t)"""
        B, N, _ = x.shape
        x = x.reshape(B, self.n_h, self.n_w, self.in_channels, self.patch_h, self.patch_w)
        x = x.permute(0, 3, 1, 4, 2, 5)              # (B, C, n_h, pH, n_w, pW)
        return x.reshape(B, self.in_channels, self.pad_f, self.pad_t)

    # ── Forward ─────────────────────────────────────

    def forward_masked(
        self,
        x:          torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss            : 스칼라
            per_sample_loss : (B,)
            mask            : (B, N) bool
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B = x.size(0)
        target = self.patchify(x)   # (B, N, patch_dim)

        tokens = self.patch_embed(target) + self.enc_pos

        ids_keep, ids_restore = _random_mask(B, self.n_patches, mask_ratio, x.device)
        n_vis = ids_keep.size(1)

        vis = torch.gather(
            tokens, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.d_enc),
        )

        for blk in self.encoder:
            vis = blk(vis)
        vis = self.enc_norm(vis)

        pred = self._decode(vis, ids_restore, n_vis)

        mask = _make_bool_mask(ids_restore, n_vis)

        err        = (pred - target).pow(2).mean(dim=-1)
        masked_err = (err * mask.float()).sum(dim=1) \
                   / mask.float().sum(dim=1).clamp(min=1)

        return masked_err.mean(), masked_err, mask

    def _decode(
        self,
        vis:         torch.Tensor,
        ids_restore: torch.Tensor,
        n_vis:       int,
    ) -> torch.Tensor:
        B = vis.size(0)
        N = ids_restore.size(1)
        n_mask = N - n_vis

        vis_dec     = self.enc_to_dec(vis)
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x = torch.cat([vis_dec, mask_tokens], dim=1)

        x = torch.gather(
            x, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.d_dec),
        )

        x = x + self.dec_pos

        for blk in self.decoder:
            x = blk(x)
        x = self.dec_norm(x)

        return self.recon_head(x)

    # ── 이상 점수 ───────────────────────────────────

    def _score_per_patch(
        self,
        x:          torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """패치별 MSE (B, N) + bool 마스크 (B, N) 반환. top-k 집계용."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B      = x.size(0)
        target = self.patchify(x)
        tokens = self.patch_embed(target) + self.enc_pos
        ids_keep, ids_restore = _random_mask(B, self.n_patches, mask_ratio, x.device)
        n_vis  = ids_keep.size(1)
        vis    = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.d_enc))
        for blk in self.encoder:
            vis = blk(vis)
        vis  = self.enc_norm(vis)
        pred = self._decode(vis, ids_restore, n_vis)
        mask = _make_bool_mask(ids_restore, n_vis)
        err  = (pred - target).pow(2).mean(dim=-1)  # (B, N)
        return err, mask

    @torch.no_grad()
    def anomaly_score(
        self,
        x:          torch.Tensor,
        n_eval:     int          = 10,
        mask_ratio: float | None = None,
        topk_ratio: float        = 1.0,
    ) -> torch.Tensor:
        """(B, 4, F, T) → (B,) 이상 점수."""
        scores = torch.zeros(x.size(0), device=x.device)
        if topk_ratio >= 1.0:
            for _ in range(n_eval):
                _, per_sample, _ = self.forward_masked(x, mask_ratio)
                scores += per_sample
        else:
            for _ in range(n_eval):
                err, mask = self._score_per_patch(x, mask_ratio)
                masked_err = err * mask.float()
                n_masked = int(mask[0].sum().item())
                k_num    = max(1, int(n_masked * topk_ratio))
                topk_vals, _ = masked_err.topk(k_num, dim=1, largest=True)
                scores += topk_vals.mean(dim=1)
        return scores / n_eval

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, F, T) → (B, d_enc) 평균 풀링 특징 벡터."""
        target = self.patchify(x)
        tokens = self.patch_embed(target) + self.enc_pos
        for blk in self.encoder:
            tokens = blk(tokens)
        return self.enc_norm(tokens).mean(dim=1)

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        return self.forward_masked(x, mask_ratio)


# ─────────────────────────────────────────────
# 통합 래퍼
# ─────────────────────────────────────────────

class OrbitMAE(nn.Module):
    """
    1D + 스펙트로그램 MAE 통합 래퍼.

    두 스트림을 공동 학습하고, 가중 합산으로 통합 이상 점수를 반환합니다.

    이상 점수 = alpha × score_1d + (1 - alpha) × score_spec
    alpha 기본값 0.3 (spec에 70% 가중치 — 진동 주파수 도메인 특성 반영)

    손실 = loss_1d + spec_loss_weight × loss_spec
    spec_loss_weight=100.0 : 패치 차원 차이(1D 1000 vs spec 256)로 인한
                              손실 규모 불균형 보정

    spec_mask_ratio=0.85 : spec 브랜치에 더 어려운 마스킹 과제 부여
                           (1D: 75% 유지, spec: 85%)

    use_spec=False 시 1D 스트림만 사용 (스펙트로그램 불필요 환경).
    """

    def __init__(
        self,
        use_spec:         bool        = True,
        alpha:            float       = 0.5,
        spec_loss_weight: float       = 100.0,  # spec 손실 스케일 보정 (1D 대비 규모 맞춤)
        spec_mask_ratio:  float | None = 0.85,   # spec 브랜치 전용 마스킹 비율 (None→ mask_ratio 공유)
        # 1D 브랜치 파라미터
        in_ch_1d:         int   = IN_CH_1D,
        seq_len:          int   = SEQ_LEN,
        patch_size:       int   = PATCH_SIZE_1D,
        # 스펙트로그램 브랜치 파라미터
        in_ch_spec:       int   = IN_CH_SPEC,
        f_bins:           int   = SPEC_F_BINS,
        t_frames:         int   = SPEC_T_FRAMES,
        patch_h:          int   = SPEC_PATCH_H,
        patch_w:          int   = SPEC_PATCH_W,
        # 공통 Transformer 파라미터
        d_enc:            int   = D_ENC,
        d_dec:            int   = D_DEC,
        n_enc:            int   = N_ENC_LAYERS,
        n_dec:            int   = N_DEC_LAYERS,
        mask_ratio:       float = MASK_RATIO,
        dropout:          float = DROPOUT,
    ):
        super().__init__()
        self.use_spec         = use_spec
        self.alpha            = alpha
        self.spec_loss_weight = spec_loss_weight
        # spec_mask_ratio가 None이면 1D와 동일하게 유지
        self.spec_mask_ratio  = spec_mask_ratio if spec_mask_ratio is not None else mask_ratio

        self.branch_1d = OrbitMAE1D(
            in_channels=in_ch_1d, seq_len=seq_len, patch_size=patch_size,
            d_enc=d_enc, d_dec=d_dec,
            n_enc_layers=n_enc, n_dec_layers=n_dec,
            mask_ratio=mask_ratio, dropout=dropout,
        )

        if use_spec:
            self.branch_spec = OrbitMAESpec(
                in_channels=in_ch_spec, f_bins=f_bins, t_frames=t_frames,
                patch_h=patch_h, patch_w=patch_w,
                d_enc=d_enc, d_dec=d_dec,
                n_enc_layers=n_enc, n_dec_layers=n_dec,
                mask_ratio=self.spec_mask_ratio, dropout=dropout,
            )

    def forward(
        self,
        x_1d:       torch.Tensor,
        x_spec:     torch.Tensor | None = None,
        mask_ratio: float | None        = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss       : 스칼라 총 손실 (역전파용)
                         = loss_1d + spec_loss_weight × loss_spec
            loss_1d    : 스칼라 1D 손실 (raw, 가중치 미적용)
            loss_spec  : 스칼라 스펙트로그램 손실 (raw, 가중치 미적용)
        """
        loss_1d, _, _ = self.branch_1d.forward_masked(x_1d, mask_ratio)

        if self.use_spec and x_spec is not None:
            # spec 브랜치는 self.spec_mask_ratio 사용 (None → branch_spec.mask_ratio 기본값)
            loss_spec, _, _ = self.branch_spec.forward_masked(x_spec, None)
            return loss_1d + self.spec_loss_weight * loss_spec, loss_1d, loss_spec

        return loss_1d, loss_1d, torch.zeros(1, device=x_1d.device)

    @torch.no_grad()
    def anomaly_score(
        self,
        x_1d:       torch.Tensor,
        x_spec:     torch.Tensor | None = None,
        n_eval:     int          = 10,
        mask_ratio: float | None = None,
        alpha:      float | None = None,
        topk_ratio: float        = 1.0,
    ) -> torch.Tensor:
        """
        통합 이상 점수.

        Args:
            x_1d       : (B, 2, 40000)
            x_spec     : (B, 4, F, T) or None
            n_eval     : Monte Carlo 반복 횟수
            alpha      : 1D 가중치 (None이면 self.alpha 사용)
            topk_ratio : 오차 상위 K% 패치 사용 (1.0=전체 평균, 0.1=상위 10%)

        Returns:
            score : (B,) 이상 점수
        """
        if alpha is None:
            alpha = self.alpha

        score_1d = self.branch_1d.anomaly_score(x_1d, n_eval, mask_ratio, topk_ratio)

        if self.use_spec and x_spec is not None:
            score_spec = self.branch_spec.anomaly_score(x_spec, n_eval, None, topk_ratio)
            return alpha * score_1d + (1.0 - alpha) * score_spec

        return score_1d
