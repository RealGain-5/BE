from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import Flowable
import os

OUTPUT = r"C:\Users\yunha\Desktop\rcp_5th\docs\RCPVMS_인수인계.pdf"

# ── 폰트 등록 ──────────────────────────────────────
FONT_DIR = r"C:\Windows\Fonts"
fonts = {
    "Malgun": "malgun.ttf",
    "MalgunBold": "malgunbd.ttf",
}
for name, fname in fonts.items():
    path = os.path.join(FONT_DIR, fname)
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))

BODY_FONT = "Malgun"
BOLD_FONT = "MalgunBold"
MONO_FONT = "Malgun"

# ── 색상 ───────────────────────────────────────────
C_TITLE    = colors.HexColor("#1F4E79")
C_H1       = colors.HexColor("#1F4E79")
C_H2       = colors.HexColor("#2E75B6")
C_H3       = colors.HexColor("#404040")
C_HDR_BG   = colors.HexColor("#1F4E79")
C_HDR_FG   = colors.white
C_SUB_BG   = colors.HexColor("#D6E4F0")
C_ALT_BG   = colors.HexColor("#F5F9FF")
C_CODE_BG  = colors.HexColor("#F4F4F4")
C_BORDER   = colors.HexColor("#CCCCCC")
C_GRAY     = colors.HexColor("#5A5A5A")
C_RED      = colors.HexColor("#C00000")
C_RED_BG   = colors.HexColor("#FFF0F0")

# ── 스타일 ─────────────────────────────────────────
def make_styles():
    return {
        "title": ParagraphStyle("title", fontName=BOLD_FONT, fontSize=26,
            textColor=C_TITLE, alignment=1, spaceAfter=6, leading=32),
        "subtitle": ParagraphStyle("subtitle", fontName=BODY_FONT, fontSize=13,
            textColor=C_GRAY, alignment=1, spaceAfter=4, leading=18),
        "h1": ParagraphStyle("h1", fontName=BOLD_FONT, fontSize=16,
            textColor=C_H1, spaceBefore=16, spaceAfter=6, leading=22,
            borderPadding=(0,0,2,0)),
        "h2": ParagraphStyle("h2", fontName=BOLD_FONT, fontSize=13,
            textColor=C_H2, spaceBefore=12, spaceAfter=4, leading=18),
        "h3": ParagraphStyle("h3", fontName=BOLD_FONT, fontSize=11,
            textColor=C_H3, spaceBefore=8, spaceAfter=3, leading=16),
        "body": ParagraphStyle("body", fontName=BODY_FONT, fontSize=10,
            spaceAfter=4, leading=15),
        "bullet": ParagraphStyle("bullet", fontName=BODY_FONT, fontSize=10,
            leftIndent=16, bulletIndent=0, spaceAfter=3, leading=15),
        "code": ParagraphStyle("code", fontName=MONO_FONT, fontSize=8.5,
            backColor=C_CODE_BG, leftIndent=12, rightIndent=8,
            leading=14, spaceAfter=1, spaceBefore=1, textColor=colors.HexColor("#2E2E2E")),
        "note": ParagraphStyle("note", fontName=BODY_FONT, fontSize=9,
            textColor=C_GRAY, leftIndent=8, spaceAfter=3, leading=13),
    }

S = make_styles()
W, H = A4
MARGIN = 20 * mm
CONTENT_W = W - 2 * MARGIN

# ── 헬퍼 ───────────────────────────────────────────
def sp(h=4): return Spacer(1, h * mm)

def h1(text): return Paragraph(text, S["h1"])
def h2(text): return Paragraph(text, S["h2"])
def h3(text): return Paragraph(text, S["h3"])
def body(text): return Paragraph(text, S["body"])
def note(text): return Paragraph(f"※ {text}", S["note"])
def bul(text): return Paragraph(f"• {text}", S["bullet"])

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4)

def code_block(lines):
    out = []
    for line in lines:
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(Paragraph(safe if safe.strip() else " ", S["code"]))
    return out

def make_table(header, rows, col_widths, alt=True):
    data = [header] + rows
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_HDR_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_HDR_FG),
        ("FONTNAME",   (0, 0), (-1, 0), BOLD_FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("FONTNAME",   (0, 1), (-1, -1), BODY_FONT),
        ("GRID",       (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, C_ALT_BG] if alt else [colors.white]),
        ("BACKGROUND", (0, 1), (0, -1), C_SUB_BG),
        ("FONTNAME",   (0, 1), (0, -1), BOLD_FONT),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("WORDWRAP",      (0, 0), (-1, -1), True),
    ]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style))
    return tbl

# ── 페이지 번호 콜백 ──────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY_FONT, 8)
    canvas.setFillColor(C_GRAY)
    canvas.drawCentredString(W / 2, 12 * mm, f"— {doc.page} —")
    canvas.setFont(BODY_FONT, 8)
    canvas.drawRightString(W - MARGIN, H - 12 * mm, "RCPVMS 프로젝트 인수인계 문서")
    canvas.restoreState()

# ── 문서 내용 조립 ────────────────────────────────
story = []

# 표지
story += [
    sp(20),
    Paragraph("RCPVMS", S["title"]),
    Paragraph("Reactor Coolant Pump Vibration Monitoring System", S["subtitle"]),
    Paragraph("프로젝트 인수인계 문서", ParagraphStyle("sub2", fontName=BOLD_FONT, fontSize=15,
        textColor=C_TITLE, alignment=1, spaceAfter=4)),
    sp(4),
    HRFlowable(width="60%", thickness=1.5, color=C_TITLE, hAlign="CENTER"),
    sp(4),
    Paragraph("2026년 5월", ParagraphStyle("date", fontName=BODY_FONT, fontSize=10,
        textColor=C_GRAY, alignment=1)),
    PageBreak(),
]

# ── 1. 프로젝트 개요 ──────────────────────────────
story += [h1("1. 프로젝트 개요 및 목적"), hr(), sp(1)]
story += [h2("1.1 프로젝트 정의"), sp(1)]
story.append(make_table(
    ["항목", "내용"],
    [
        ["시스템 명칭", "RCPVMS (Reactor Coolant Pump Vibration Monitoring System)"],
        ["형태", "Electron 기반 Windows 데스크톱 애플리케이션"],
        ["목적", "원자력 발전소 냉각재 펌프(RCP) 진동 신호 시각화 및 AI 이상 탐지"],
        ["대상 데이터", "RCPVMS 전용 .BIN 바이너리 파일, DMD 계측 장비 출력 파일"],
    ],
    [CONTENT_W * 0.28, CONTENT_W * 0.72]
))
story += [sp(2), h2("1.2 핵심 기능 목록"), sp(1)]
story.append(make_table(
    ["탭", "기능"],
    [
        ["앙상블 분석", "ResNet18 + CNN1D 앙상블로 정상/이상 4-class 분류 + OOD 탐지"],
        ["MAE 분석", "Masked Autoencoder 기반 재구성 오차 이상 탐지"],
        ["DMD 분석", "DMD 파일 → RCPVMS BIN 변환 + 궤도 뷰어"],
        ["RCPVMS 뷰어", "BIN 파일 궤도(orbit) 그리드 시각화 (단일/배치)"],
    ],
    [CONTENT_W * 0.28, CONTENT_W * 0.72]
))
story += [sp(2), h2("1.3 모니터링 대상"), sp(1)]
story += [
    bul("RCP 4개: RCP1A / RCP1B / RCP2A / RCP2B (NIMS 표준 명칭)"),
    bul("채널 타입: 변위(orbit, ch_type=1), 가속도(ch_type=0), Keyphasor(ch_type=2)"),
    sp(2),
]

# ── 2. 기술 스택 ──────────────────────────────────
story += [PageBreak(), h1("2. 기술 스택"), hr(), sp(1)]
story += [h2("2.1 Frontend / Desktop"), sp(1)]
story.append(make_table(
    ["항목", "버전", "내용"],
    [
        ["Electron", "^38.1.2", "메인 프로세스 + 브라우저 창 관리"],
        ["electron-vite", "^4.0.1", "빌드/번들링/HMR"],
        ["React", "JSX", "렌더러 UI"],
        ["TypeScript", "^5.9.2", "Main/Preload 계층"],
        ["better-sqlite3", "^12.5.0", "로컬 DB (로그, 인증)"],
        ["ExcelJS", "^4.4.0", "결과 Excel 내보내기"],
        ["electron-store", "^6.0.1", "앱 설정 영속화"],
    ],
    [CONTENT_W * 0.28, CONTENT_W * 0.22, CONTENT_W * 0.50]
))
story += [sp(2), h2("2.2 Backend (Python 데몬)"), sp(1)]
story.append(make_table(
    ["항목", "버전", "내용"],
    [
        ["Python", "3.10+ (3.12 권장)", "가상환경: venv/"],
        ["PyTorch", ">=2.0.0", "CPU; GPU는 CUDA 버전으로 교체"],
        ["torchvision", ">=0.15.0", "이미지 변환"],
        ["NumPy", ">=1.24.0", "수치 연산"],
        ["Pillow", ">=10.0.0", "이미지 생성/처리"],
        ["SciPy", ">=1.10.0", "신호 처리, 필터링"],
        ["Matplotlib", ">=3.7.0", "축 레이블 렌더링"],
    ],
    [CONTENT_W * 0.28, CONTENT_W * 0.22, CONTENT_W * 0.50]
))
story += [sp(2), h2("2.3 IPC 통신"), sp(1)]
story += [
    bul("Electron ipcMain / ipcRenderer + contextBridge"),
    bul("Python 프로세스와 stdin/stdout JSON 스트리밍 방식으로 통신"),
    sp(2),
]

# ── 3. 데이터 흐름 ────────────────────────────────
story += [PageBreak(), h1("3. 데이터 흐름 (Data Flow)"), hr(), sp(1)]
story += [h2("3.1 전체 아키텍처 흐름"), sp(1)]
story += code_block([
    "Renderer (React UI)  window.api.xxx() 호출",
    "    | ipcRenderer.invoke()",
    "Preload (contextBridge)  window.api 객체 노출",
    "    | IPC channel",
    "Main Process (index.ts + services)",
    "    ipcMain.handle() -> pythonService.runXxx()",
    "    | PythonDaemonPool",
    "Worker 1 (stdin/stdout) ... Worker N (stdin/stdout)",
    "    |",
    "inference_daemon.py",
    "    명령 수신 -> 해당 모듈 호출 -> JSON 응답",
    "    캐시: _rcpvms_header_cache / _rcpvms_orbit_cache",
    "    |",
    "    rcpvms_parser  preprocess.py  model_loader.py",
    "    dmd_parser     infer_resnet   model_mae.py",
])
story += [sp(2), h2("3.2 AI 추론 데이터 흐름"), sp(1)]
story += code_block([
    "BIN 파일",
    "  +-> rcpvms_parser.py (헤더 파싱 + 채널 추출)",
    "        +-> preprocess.py",
    "              +- extract_rcp_xy_from_bin()  <- X/Y 변위 채널",
    "              +- estimate_1x_freq()          <- 1X 회전 주파수",
    "              +- _make_display_pil()         <- 궤도 이미지",
    "                    | [filter_mode: raw / 1x / 2x / broadband / overlay]",
    "        infer_resnet_None.py  +  train_1d_cnn.py",
    "        (ResNet18 멀티스케일)    (OrbitCNN1D)",
    "              | probabilities        | probabilities",
    "              +--------- 앙상블 가중합 ---------+",
    "                    |",
    "              OOD 판정 (TV Distance + max_conf)",
    "                    |",
    "         normal / abnormal_typeA~C / unknown_abnormal",
    "                    |",
    "         GradCAM + Integrated Gradients 시각화",
])
story += [sp(2), h2("3.3 MAE 이상 탐지 흐름"), sp(1)]
story += code_block([
    "BIN 파일",
    "  +-> Stage 1: 슬라이딩 윈도우 배치 스윕",
    "        -> 재구성 오차 점수 계산 (1D + Spectral)",
    "        -> 최고 점수 윈도우 선정",
    "  +-> Stage 2: 선정 윈도우 재평가 (n_eval=10)",
    "        -> OR 로직: (score_1d > threshold_1d) OR (score_spec > threshold_spec)",
    "        -> 최종 판정: normal / abnormal",
])
story += [sp(2), h2("3.4 궤도 뷰어 데이터 흐름"), sp(1)]
story += code_block([
    "BIN 파일 선택",
    "  +-> rcpvms-info        파일 메타 (채널 수, 샘플링 주파수, 총 시간)",
    "  +-> rcpvms-orbit       모든 RCP/시간 윈도우 그리드 정보",
    "  +-> rcpvms-orbit-multi (배치 API)",
    "        IntersectionObserver: 뷰포트 진입 시만 요청",
    "        썸네일 96px (PIL _draw_crosshair, matplotlib 생략)",
    "        모달 256px (render_with_axes — 축 레이블 포함)",
])
story.append(sp(2))

# ── 4. 컴포넌트 맵 ────────────────────────────────
story += [PageBreak(), h1("4. 기능 개요도 (Component Map)"), hr(), sp(1)]
story += code_block([
    "App.jsx (탭 라우터)",
    "  +-- [앙상블 분석] ModelInference.jsx",
    "  |     +-- AnalysisModeLayout.jsx (공통 레이아웃)",
    "  |     |     +-- SingleFileMode.jsx",
    "  |     |     +-- BatchFileList.jsx",
    "  |     |     +-- BatchProgressBar.jsx",
    "  |     |     +-- BatchActionButtons.jsx",
    "  |     |     +-- BatchResultList.jsx",
    "  |     |     +-- ConcurrencySelector.jsx",
    "  |     |     +-- ErrorDisplay.jsx",
    "  |     +-- labelStrategies.jsx (ensemble 판정 렌더링)",
    "  |",
    "  +-- [MAE 분석] MAEAnalysis.jsx",
    "  |     +-- AnalysisModeLayout.jsx (동일 공유)",
    "  |     +-- labelStrategies.jsx (mae 판정 렌더링)",
    "  |",
    "  +-- [DMD 분석] DmdOrbitViewer.jsx",
    "  |     +-- SubTabNav.jsx",
    "  |     +-- [서브탭1] DMD 변환: FileOperationFlow.jsx",
    "  |     +-- [서브탭2] BIN 궤도 뷰어: FileOperationFlow.jsx -> OrbitGrid.jsx",
    "  |",
    "  +-- [RCPVMS 뷰어] RcpvmsOrbitViewer.jsx",
    "        +-- FileOperationFlow.jsx",
    "        +-- OrbitControls.jsx (필터 모드 / window_sec / axis_lim)",
    "        +-- OrbitGrid.jsx (IntersectionObserver + 배치 API)",
])
story.append(sp(2))

# ── 5. 환경 세팅 ──────────────────────────────────
story += [PageBreak(), h1("5. 프로그램 다운로드 및 환경 세팅"), hr(), sp(1)]
story += [h2("5.1 사전 준비"), sp(1)]
story.append(make_table(
    ["소프트웨어", "권장 버전", "비고"],
    [
        ["Git", "최신", "https://git-scm.com"],
        ["Node.js", "20.x LTS 이상", "https://nodejs.org"],
        ["Python", "3.10 이상 (3.12 권장)", '"Add to PATH" 필수 체크'],
        ["Visual Studio Code", "최신", "선택 사항"],
    ],
    [CONTENT_W * 0.28, CONTENT_W * 0.22, CONTENT_W * 0.50]
))
story += [sp(2), h2("5.2 코드 다운로드"), sp(1)]
story += code_block([
    "cd D:\\projects",
    "git clone https://github.com/RealGain-5/BE.git rcp_5th",
    "cd rcp_5th",
])
story += [sp(2), h2("5.3 Node.js 의존성 설치"), sp(1)]
story += code_block([
    "npm install",
    "cd src/renderer && npm install && cd ../..",
])
story += [note("npm install 후 postinstall 스크립트가 better-sqlite3를 Electron 버전에 맞게 자동 재빌드합니다."),
          note("오류 발생 시 수동 실행: npx @electron/rebuild"), sp(2)]
story += [h2("5.4 Python 가상환경 세팅"), sp(1)]
story += code_block([
    ".\\venv\\Scripts\\Activate.ps1",
    "# PowerShell 실행 정책 오류 시:",
    "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
    "",
    "python -m pip install --upgrade pip",
    "pip install -r python/requirements.txt",
])
story += [note("GPU 사용 시: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"), sp(2)]
story += [h2("5.5 AI 모델 파일 배치"), sp(1)]
story += code_block([
    "python/model/",
    "  +-- resnet18_orbit_multiscale.pth   앙상블 ResNet18",
    "  +-- orbit_cnn1d.pth                 앙상블 CNN1D",
    "  +-- orbit_mae.pth                   MAE 이상 탐지",
])
story += [note("모델 파일이 없으면 담당자에게 요청하십시오."), sp(2)]
story += [h2("5.6 설정 파일 확인"), sp(1)]
story += code_block([
    "python/ensemble_config.json   앙상블 가중치 / OOD 임계값",
    "python/mae_config.json        MAE 임계값",
    "",
    "ensemble_config.json 기본값:",
    '  { "resnet_weight": 0.5, "cnn1d_weight": 0.5,',
    '    "ood_threshold": 0.70, "tv_threshold": 0.30 }',
])
story.append(sp(2))

# ── 6. Dev 모드 실행 ──────────────────────────────
story += [PageBreak(), h1("6. 개발 모드 실행 방법"), hr(), sp(1)]
story += [h2("6.1 개발 모드 실행"), sp(1)]
story += code_block(["npm run dev"])
story += [sp(1), body("내부 동작 순서:"),
    bul("Vite 개발 서버(렌더러) 시작"),
    bul("Electron 메인 프로세스 시작"),
    bul("BrowserWindow 생성 → 렌더러 URL 로드"),
    bul("Python 데몬 자동 시작: venv/Scripts/python python/inference_daemon.py"),
    sp(2)]
story += [h2("6.2 개발 모드 확인 포인트"), sp(1)]
story.append(make_table(
    ["확인 사항", "방법"],
    [
        ["Python 데몬 준비 확인", "콘솔에 'model loaded successfully' 출력 대기"],
        ["DevTools 열기", "F12 키 (개발 모드에서만 활성)"],
        ["메인 프로세스 로그", "Electron 터미널 창 확인"],
        ["렌더러 로그", "DevTools Console 탭 확인"],
    ],
    [CONTENT_W * 0.40, CONTENT_W * 0.60]
))
story += [sp(2), h2("6.3 TypeScript 타입 체크"), sp(1)]
story += code_block(["npm run typecheck"])
story.append(sp(2))

# ── 7. 프로덕션 빌드 ──────────────────────────────
story += [h1("7. 프로덕션 빌드 방법"), hr(), sp(1)]
story += code_block([
    "# Windows NSIS 설치 파일 빌드",
    "npm run build:win",
    "",
    "# 언팩 빌드 (테스트용)",
    "npm run build:unpack",
])
story += [sp(1), body("빌드 결과물: dist/ 폴더에 .exe 설치 파일 생성"), sp(2)]

# ── 8. 파일 위치 참조 ────────────────────────────
story += [PageBreak(), h1("8. 주요 파일 위치 빠른 참조"), hr(), sp(1)]
story.append(make_table(
    ["목적", "파일 경로"],
    [
        ["탭 추가/수정", "src/renderer/src/utils/tabRegistry.js"],
        ["IPC 핸들러 추가", "src/main/index.ts"],
        ["Python 서비스 메서드 추가", "src/main/services/pythonService.ts"],
        ["preload API 노출", "src/preload/index.ts"],
        ["Python 명령 처리", "python/inference_daemon.py"],
        ["신호 처리/필터링", "python/preprocess.py"],
        ["AI 모델 정의", "python/model_mae.py, python/infer_resnet_None.py"],
        ["앙상블 설정", "python/ensemble_config.json"],
    ],
    [CONTENT_W * 0.35, CONTENT_W * 0.65]
))
story.append(sp(2))

# ── 9. 미완료 작업 ────────────────────────────────
story += [h1("9. 미완료 작업 및 인수인계 체크리스트"), hr(), sp(1)]

todo_data = [["항목", "상태", "비고"]] + [
    ["앙상블 모델 재학습", "미완료", "OE + 복합 지표 코드 완성, 실제 재학습 필요"],
    ["일반화 평가", "미완료", "다른 날짜/조건 BIN 파일 테스트 필요"],
    ["DMD 실제 파일 변환 검증", "미완료", "5.1GB DMD 파일 변환 테스트 필요"],
    ["변환 BIN → 학습 파이프라인 투입", "미완료", "변환된 BIN 파일로 CNN 재학습 필요"],
]
todo_style = [
    ("BACKGROUND",    (0, 0), (-1, 0), C_HDR_BG),
    ("TEXTCOLOR",     (0, 0), (-1, 0), C_HDR_FG),
    ("FONTNAME",      (0, 0), (-1, 0), BOLD_FONT),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("FONTNAME",      (0, 1), (-1, -1), BODY_FONT),
    ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
    ("BACKGROUND",    (0, 1), (0, -1), C_SUB_BG),
    ("FONTNAME",      (0, 1), (0, -1), BOLD_FONT),
    ("BACKGROUND",    (1, 1), (1, -1), C_RED_BG),
    ("TEXTCOLOR",     (1, 1), (1, -1), C_RED),
    ("FONTNAME",      (1, 1), (1, -1), BOLD_FONT),
    ("ROWBACKGROUNDS",(2, 1), (2, -1), [colors.white, C_ALT_BG]),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
]
t = Table(todo_data, colWidths=[CONTENT_W * 0.38, CONTENT_W * 0.14, CONTENT_W * 0.48], repeatRows=1)
t.setStyle(TableStyle(todo_style))
story.append(t)
story.append(sp(2))

# ── 문서 빌드 ─────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=18 * mm, bottomMargin=18 * mm,
    title="RCPVMS 프로젝트 인수인계 문서",
    author="Joe",
    subject="RCPVMS Handover Document",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF created: {OUTPUT}")
