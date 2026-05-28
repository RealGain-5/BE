# -*- coding: utf-8 -*-
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
import os

OUTPUT = r"C:\Users\yunha\Desktop\rcp_5th\docs\RCPVMS_초기세팅가이드.pdf"
FONT_DIR = r"C:\Windows\Fonts"

for name, fname in {"Malgun": "malgun.ttf", "MalgunBold": "malgunbd.ttf"}.items():
    path = os.path.join(FONT_DIR, fname)
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))

BF = "Malgun"
BBF = "MalgunBold"

C_TITLE   = colors.HexColor("#1F4E79")
C_H1      = colors.HexColor("#1F4E79")
C_H2      = colors.HexColor("#2E75B6")
C_HDR_BG  = colors.HexColor("#1F4E79")
C_SUB_BG  = colors.HexColor("#D6E4F0")
C_ALT_BG  = colors.HexColor("#F5F9FF")
C_CODE_BG = colors.HexColor("#F4F4F4")
C_WARN_BG = colors.HexColor("#FFF8E1")
C_TIP_BG  = colors.HexColor("#E8F5E9")
C_ERR_BG  = colors.HexColor("#FFEBEE")
C_INFO_BG = colors.HexColor("#E3F2FD")
C_BORDER  = colors.HexColor("#CCCCCC")
C_GRAY    = colors.HexColor("#5A5A5A")
C_RED     = colors.HexColor("#C00000")
C_RED_BG  = colors.HexColor("#FFF0F0")

W, H = A4
MARGIN = 20 * mm
CW = W - 2 * MARGIN

def S(name, **kw):
    base = {
        "title":    dict(fontName=BBF, fontSize=26, textColor=C_TITLE, alignment=1, spaceAfter=4, leading=32),
        "sub":      dict(fontName=BF,  fontSize=12, textColor=C_GRAY,  alignment=1, spaceAfter=4, leading=18),
        "h1":       dict(fontName=BBF, fontSize=16, textColor=C_H1, spaceBefore=14, spaceAfter=5, leading=22),
        "h2":       dict(fontName=BBF, fontSize=13, textColor=C_H2, spaceBefore=10, spaceAfter=4, leading=18),
        "h3":       dict(fontName=BBF, fontSize=11, textColor=colors.HexColor("#404040"), spaceBefore=7, spaceAfter=3, leading=16),
        "body":     dict(fontName=BF,  fontSize=9.5, spaceAfter=4, leading=15),
        "bul":      dict(fontName=BF,  fontSize=9.5, leftIndent=14, spaceAfter=3, leading=15),
        "code":     dict(fontName=BF,  fontSize=8.5, backColor=C_CODE_BG, leftIndent=10, rightIndent=6, leading=14, spaceAfter=1, spaceBefore=1),
        "callout":  dict(fontName=BF,  fontSize=9,   leftIndent=10, spaceAfter=2, leading=14),
        "note":     dict(fontName=BF,  fontSize=8.5, textColor=C_GRAY, leftIndent=8, spaceAfter=3, leading=13, italics=True),
    }.get(name, {})
    base.update(kw)
    return ParagraphStyle(name + str(id(kw)), **base)

def sp(h=3): return Spacer(1, h * mm)
def h1(t): return Paragraph(t, S("h1"))
def h2(t): return Paragraph(t, S("h2"))
def h3(t): return Paragraph(t, S("h3"))
def body(t): return Paragraph(t, S("body"))
def note(t): return Paragraph(f"※ {t}", S("note"))
def bul(t): return Paragraph(f"• {t}", S("bul"))
def code(t): return Paragraph(t if t.strip() else " ", S("code"))
def codes(lines): return [code(l) for l in lines]

def callout(icon, label, lines, bg):
    out = []
    first = True
    for line in lines:
        style = ParagraphStyle(f"co{id(line)}", fontName=BF, fontSize=9,
            backColor=bg, leftIndent=10, rightIndent=6, leading=14,
            spaceBefore=(3 if first else 0), spaceAfter=0)
        prefix = f"<b>{icon} {label}</b>  " if first else "    "
        txt = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(Paragraph(f"{prefix}{txt}", style))
        first = False
    out.append(Spacer(1, 3 * mm))
    return out

def make_table(header, rows, widths, first_col_bold=True):
    data = [header] + rows
    ts = [
        ("BACKGROUND",    (0,0),  (-1,0),  C_HDR_BG),
        ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
        ("FONTNAME",      (0,0),  (-1,0),  BBF),
        ("FONTSIZE",      (0,0),  (-1,-1), 8.5),
        ("FONTNAME",      (0,1),  (-1,-1), BF),
        ("GRID",          (0,0),  (-1,-1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS",(0,1),  (-1,-1), [colors.white, C_ALT_BG]),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),  (-1,-1), 5),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
        ("LEFTPADDING",   (0,0),  (-1,-1), 6),
        ("RIGHTPADDING",  (0,0),  (-1,-1), 6),
        ("WORDWRAP",      (0,0),  (-1,-1), True),
    ]
    if first_col_bold:
        ts += [
            ("BACKGROUND", (0,1), (0,-1), C_SUB_BG),
            ("FONTNAME",   (0,1), (0,-1), BBF),
        ]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(ts))
    return t

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(BF, 8)
    canvas.setFillColor(C_GRAY)
    canvas.drawCentredString(W/2, 10*mm, f"— {doc.page} —")
    canvas.drawRightString(W - MARGIN, H - 11*mm, "RCPVMS 초기 세팅 가이드")
    canvas.restoreState()

story = []

# 표지
story += [
    sp(18),
    Paragraph("RCPVMS", S("title")),
    Paragraph("초기 세팅 가이드 및 별도 제공 파일 목록", S("sub")),
    Paragraph("Git 기반 프로젝트 설치 · 환경 구성 완전 가이드", ParagraphStyle("s2", fontName=BBF, fontSize=14, textColor=C_H2, alignment=1, spaceAfter=4)),
    sp(4),
    HRFlowable(width="60%", thickness=1.5, color=C_TITLE, hAlign="CENTER"),
    sp(4),
    Paragraph("대상 독자: Git 및 개발 환경 초보자  |  2026년 5월",
        ParagraphStyle("date", fontName=BF, fontSize=10, textColor=C_GRAY, alignment=1)),
    PageBreak(),
]

# 문서 구성
story += [h1("문서 구성"), sp(1)]
story.append(make_table(
    ["섹션", "내용"],
    [["1","이 문서를 읽기 전에 — 기본 개념 설명"],
     ["2","STEP 1  필수 소프트웨어 설치"],
     ["3","STEP 2  Git으로 코드 내려받기"],
     ["4","STEP 3  Node.js 패키지 설치"],
     ["5","STEP 4  Python 가상환경 구성"],
     ["6","STEP 5  앱 실행 확인"],
     ["7","별도 제공 파일 목록 및 배치 방법"],
     ["8","자주 발생하는 오류와 해결 방법"]],
    [CW*0.08, CW*0.92], first_col_bold=False
))
story += [sp(2), PageBreak()]

# 1. 기본 개념
story += [h1("1. 이 문서를 읽기 전에 — 기본 개념"), sp(1)]
story += [h2("1.1 Git이란?"), sp(1)]
story += [body("Git은 소스 코드를 관리하고 공유하는 도구입니다. 마치 문서 작업 시 '버전 관리(v1, v2, 최종, 진짜최종…)'를 체계적으로 해 주는 프로그램으로 생각하면 됩니다."),
          body("GitHub(https://github.com)는 Git으로 관리되는 코드를 인터넷에 올려두는 저장소 서비스입니다. 이 프로젝트 코드도 GitHub에 저장되어 있으며, Git 명령어로 내 컴퓨터에 내려받을 수 있습니다."), sp(1)]
story += [h2("1.2 터미널(명령 프롬프트)이란?"), sp(1)]
story += [body("터미널은 마우스 대신 키보드로 컴퓨터에 명령을 내리는 창입니다. 이 문서에서는 Windows의 PowerShell을 사용합니다.")]
story += callout("▶", "PowerShell 여는 방법",
    ["키보드에서 Windows 키 + R 을 동시에 누릅니다.",
     "실행 창에 powershell 을 입력하고 Enter 를 누릅니다.",
     "파란 창이 뜨면 성공입니다."], C_INFO_BG)
story += [h2("1.3 명령어 입력 방법"), sp(1)]
story += [body("이 문서에서 회색 박스 안의 내용은 터미널에 그대로 입력하는 명령어입니다. 한 줄씩 입력하고 Enter 키를 눌러 실행합니다.")]
story += callout("⚠", "주의",
    ["명령어 앞의 번호(1. 2. 3.)는 순서 표시일 뿐, 입력하지 않습니다.",
     "# 기호로 시작하는 줄은 설명(주석)이므로 입력하지 않아도 됩니다."], C_WARN_BG)

# 2. 소프트웨어 설치
story += [PageBreak(), h1("2. STEP 1 — 필수 소프트웨어 설치"), sp(1)]
story += [body("아래 4가지 프로그램을 순서대로 설치합니다. 이미 설치되어 있는 항목은 건너뛰어도 됩니다."), sp(1)]
story.append(make_table(
    ["프로그램","용도","확인 명령어"],
    [["Git","코드 저장소에서 프로젝트를 내려받음","git --version"],
     ["Node.js v20 LTS","앱의 JavaScript 런타임 환경","node --version"],
     ["Python 3.12","AI 모델 실행 환경","python --version"],
     ["Visual Studio Code","코드 편집기 (선택 사항)","code --version"]],
    [CW*0.26, CW*0.46, CW*0.28]
))
story += [sp(2), h2("2.1 Git 설치"), sp(1)]
story += [body("① 웹 브라우저에서 https://git-scm.com 에 접속합니다."),
          body("② 화면 중앙의 [Download for Windows] 버튼을 클릭해 설치 파일을 내려받습니다."),
          body("③ 내려받은 파일을 실행하고, 모든 설치 옵션을 기본값(Next)으로 진행합니다."),
          body("④ PowerShell을 새로 열고 아래 명령어로 확인합니다.")]
story += codes(["git --version", "  결과 예시: git version 2.47.0.windows.2"])
story += [sp(2), h2("2.2 Node.js 설치"), sp(1)]
story += [body("① https://nodejs.org 에서 [LTS] 버전을 내려받아 설치합니다."),
          body("② 기본값으로 설치 완료 후 아래 명령어로 확인합니다.")]
story += codes(["node --version", "npm --version", "  결과 예시: v20.18.0 / 10.8.2"])
story += [sp(2), h2("2.3 Python 설치"), sp(1)]
story += [body("① https://www.python.org/downloads 에서 Python 3.12.x 설치 파일을 내려받습니다."),
          body("② 설치 파일을 실행합니다.")]
story += callout("★", "필수 체크 항목",
    ['설치 화면 맨 아래 "Add Python 3.12 to PATH" 체크박스를 반드시 체크하세요.',
     "이 항목을 체크하지 않으면 이후 모든 단계에서 python 명령어가 동작하지 않습니다."], C_ERR_BG)
story += [body("③ [Install Now] 를 클릭하고 설치를 완료합니다."),
          body("④ PowerShell에서 아래 명령어로 확인합니다.")]
story += codes(["python --version", "  결과 예시: Python 3.12.x"])
story.append(sp(2))

# 3. 코드 클론
story += [PageBreak(), h1("3. STEP 2 — Git으로 코드 내려받기 (클론)"), sp(1)]
story += [h2("3.1 프로젝트를 저장할 폴더 위치 결정"), sp(1)]
story += [body("코드를 저장할 폴더를 먼저 정합니다. 아래 예시에서는 D:\\projects 폴더를 사용합니다.")]
story += callout("▶", "예시 경로",
    ["D:\\\\projects\\\\rcp_5th  →  D 드라이브에 projects 폴더 안에 저장",
     "C:\\\\Users\\\\사용자이름\\\\Desktop\\\\rcp_5th  →  바탕화면에 저장"], C_INFO_BG)
story += [h2("3.2 폴더 이동 및 클론 실행"), sp(1)]
story += [body("① 저장할 상위 폴더로 이동합니다.")]
story += codes(["cd D:\\projects"])
story += callout("▶", "폴더가 없다면?",
    ["mkdir D:\\\\projects  를 먼저 입력해 폴더를 만든 후 위 명령어를 실행하세요."], C_TIP_BG)
story += [body("② GitHub에서 코드를 내 컴퓨터로 내려받습니다. (클론)")]
story += codes(["git clone https://github.com/RealGain-5/BE.git rcp_5th"])
story += [body("  Cloning into 'rcp_5th'... 메시지와 함께 파일이 내려받아집니다. 30초~2분 소요됩니다."), sp(1)]
story += [body("③ 프로젝트 폴더 안으로 이동합니다.")]
story += codes(["cd rcp_5th"])
story += [body("④ 파일 목록을 확인합니다.")]
story += codes(["ls", "  build/  docs/  python/  resources/  src/  package.json  ..."])
story.append(sp(2))

# 4. Node.js 패키지
story += [PageBreak(), h1("4. STEP 3 — Node.js 패키지 설치"), sp(1)]
story += [body("프로젝트에서 사용하는 JavaScript 라이브러리를 설치합니다. rcp_5th 폴더 안에 있는지 먼저 확인하세요."), sp(1)]
story += [h2("4.1 현재 위치 확인"), sp(1)]
story += codes(["pwd", "  # 출력 결과가 ...\\rcp_5th 로 끝나야 합니다."])
story += [sp(1), h2("4.2 루트 패키지 설치"), sp(1)]
story += codes(["npm install"])
story += [body("  수십~수백 줄의 설치 로그가 출력됩니다. 마지막에 added N packages 메시지가 나오면 성공입니다.")]
story += callout("ℹ", "자동으로 실행되는 과정",
    ["npm install 완료 후 postinstall 스크립트가 자동으로 실행됩니다.",
     "native 모듈(better-sqlite3)이 Electron 버전에 맞게 재빌드됩니다.",
     "오류 발생 시 8절 '자주 발생하는 오류' 항목을 참고하세요."], C_INFO_BG)
story += [h2("4.3 렌더러 패키지 설치"), sp(1)]
story += codes(["cd src\\renderer", "npm install", "cd ..\\.."])
story += [body("  설치 완료 후 원래 위치(rcp_5th)로 돌아옵니다."), sp(2)]

# 5. Python 가상환경
story += [PageBreak(), h1("5. STEP 4 — Python 가상환경 구성"), sp(1)]
story += [h2("5.1 가상환경이란?"), sp(1)]
story += [body("가상환경은 이 프로젝트에서만 사용하는 별도의 Python 공간입니다. 컴퓨터 전체에 영향을 주지 않고 필요한 라이브러리만 깔끔하게 설치할 수 있습니다."), sp(1)]
story += [h2("5.2 PowerShell 실행 정책 설정 (최초 1회)"), sp(1)]
story += codes(["Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"])
story += [body("  Y 또는 A 를 입력하고 Enter 를 눌러 확인합니다."), sp(1)]
story += [h2("5.3 가상환경 생성"), sp(1)]
story += codes(["python -m venv venv"])
story += [body("  완료되면 rcp_5th\\venv\\ 폴더가 생성됩니다. 30초~1분 소요됩니다."), sp(1)]
story += [h2("5.4 가상환경 활성화"), sp(1)]
story += codes([".\\venv\\Scripts\\Activate.ps1"])
story += [body("  성공하면 프롬프트 맨 앞에 (venv) 가 표시됩니다.")]
story += codes(["# 활성화 전: PS C:\\..\\rcp_5th>", "# 활성화 후: (venv) PS C:\\..\\rcp_5th>"])
story += [sp(1), h2("5.5 pip 업그레이드 및 라이브러리 설치"), sp(1)]
story += codes(["python -m pip install --upgrade pip", "pip install -r python\\requirements.txt"])
story += [body("  PyTorch 등 용량이 큰 라이브러리 포함 — 네트워크 속도에 따라 5~20분 소요됩니다.")]
story += callout("▶", "GPU 사용 시 (선택 사항)",
    ["NVIDIA GPU가 있는 경우 아래 명령어로 CUDA 버전 PyTorch를 설치하면 추론 속도가 빨라집니다.",
     "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
     "(기본 설치는 CPU 버전으로도 정상 동작합니다.)"], C_TIP_BG)
story += [h2("5.6 설치 확인"), sp(1)]
story += codes(["python -c \"import torch, numpy, PIL, scipy; print('OK')\""])
story += [body("  OK 가 출력되면 정상입니다."), sp(2)]

# 6. 앱 실행
story += [PageBreak(), h1("6. STEP 5 — 앱 실행 확인"), sp(1)]
story += [h2("6.1 모델 파일 배치 확인 (중요)"), sp(1)]
story += [body("앱을 실행하기 전에 AI 모델 파일이 올바른 위치에 있는지 반드시 확인해야 합니다.")]
story += codes(["ls python\\model"])
story += [body("  아래 3개 파일이 모두 보여야 합니다.")]
story += codes(["    resnet18_orbit_multiscale.pth   (42.7 MB)",
                "    orbit_cnn1d.pth                 ( 2.8 MB)",
                "    orbit_mae.pth                   (30.3 MB)"])
story += [body("  파일이 없다면 7절을 참고해 담당자에게 요청한 후 배치하십시오."), sp(1)]
story += [h2("6.2 개발 모드 실행"), sp(1)]
story += codes(["npm run dev"])
story += codes(["  > vite dev ...",
                "  > electron .",
                "  [Python] model loaded successfully    <- 이 메시지가 나오면 준비 완료"])
story += callout("⚠", "앱 창이 뜨지 않는다면?",
    ["터미널에서 ERROR 메시지를 확인하세요.",
     "모델 파일이 없는 경우 'FileNotFoundError' 오류가 납니다.",
     "Python 가상환경이 활성화(venv)되어 있는지 확인하세요."], C_WARN_BG)
story += [h2("6.3 정상 동작 확인"), sp(1)]
story.append(make_table(
    ["확인 항목","방법","정상 결과"],
    [["앱 창 실행","npm run dev 후 대기","Electron 창이 열림"],
     ["Python 데몬 준비","터미널 출력 확인","model loaded successfully 출력"],
     ["탭 전환","앱 상단 탭 클릭","4개 탭이 정상 전환됨"],
     ["BIN 파일 로딩","RCPVMS 뷰어 탭에서 파일 선택","궤도 이미지 표시됨"]],
    [CW*0.26, CW*0.32, CW*0.42]
))
story.append(sp(2))

# 7. 별도 제공 파일
story += [PageBreak(), h1("7. 별도 제공 파일 목록 및 배치 방법"), sp(1)]
story += [body("아래 파일들은 Git 저장소에 포함되지 않으므로 담당자에게 별도로 요청해야 합니다."), sp(1)]
story += [h2("7.1 AI 모델 체크포인트 (필수)"), sp(1)]
story += [body("AI 추론 기능을 사용하려면 아래 3개 파일이 반드시 필요합니다. 총 용량: 약 75.8 MB"), sp(1)]
story.append(make_table(
    ["파일명","크기","용도","배치 경로"],
    [["resnet18_orbit_multiscale.pth","42.7 MB","앙상블 ResNet18","python\\model\\"],
     ["orbit_cnn1d.pth","2.8 MB","앙상블 CNN1D","python\\model\\"],
     ["orbit_mae.pth","30.3 MB","MAE 이상 탐지","python\\model\\"]],
    [CW*0.38, CW*0.12, CW*0.28, CW*0.22], first_col_bold=False
))
story += [sp(1), body("배치 방법: python\\model\\ 폴더가 없으면 아래 명령어로 생성 후 파일을 복사합니다.")]
story += codes(["mkdir python\\model"])
story += [sp(1), h2("7.2 측정 데이터 파일 (기능 사용 시 필요)"), sp(1)]
story.append(make_table(
    ["파일 종류","확장자","용도","배치 경로"],
    [["RCPVMS 측정 데이터",".BIN","궤도 뷰어, 앙상블/MAE 분석","임의 경로 (앱에서 직접 선택)"],
     ["DMD 계측 데이터",".dmd","DMD 분석 탭 — BIN 변환 소스","임의 경로 (앱에서 직접 선택)"]],
    [CW*0.28, CW*0.12, CW*0.34, CW*0.26]
))
story += callout("ℹ", "참고",
    ["BIN / DMD 파일은 특정 경로에 고정하지 않아도 됩니다.",
     "앱 실행 후 각 탭에서 '파일 선택' 버튼으로 원하는 파일을 직접 불러옵니다."], C_INFO_BG)
story += [h2("7.3 .gitignore 제외 항목 전체 목록"), sp(1)]
story.append(make_table(
    ["제외 항목","이유","조치"],
    [["python/model/*.pth","파일 크기 큼 (총 ~76 MB)","7.1절 — 담당자에게 요청"],
     ["data/ 및 data/*","대용량 측정 데이터","담당자에게 별도 수령"],
     ["dmd/ 및 dmd/*","대용량 DMD 원본 파일","담당자에게 별도 수령"],
     ["venv/","Python 가상환경 (로컬 생성)","5절 절차대로 직접 생성"],
     ["node_modules/","Node 패키지 (로컬 설치)","npm install 로 자동 설치"],
     ["dist/","빌드 산출물","npm run build:win 으로 생성"]],
    [CW*0.30, CW*0.30, CW*0.40]
))
story.append(sp(2))

# 8. 오류 해결
story += [PageBreak(), h1("8. 자주 발생하는 오류와 해결 방법"), sp(1)]
story.append(make_table(
    ["오류 메시지","원인","해결 방법"],
    [["'git' is not recognized","Git PATH 미등록","Git 재설치 후 PowerShell 새로 열기"],
     ["'python' is not recognized","Python PATH 미등록","재설치 시 'Add to PATH' 체크 필수"],
     ["Activate.ps1 실행 정책 오류","스크립트 실행 차단","Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"],
     ["npm install node-gyp 오류","네이티브 모듈 빌드 실패","npx @electron/rebuild 별도 실행"],
     ["FileNotFoundError: .pth 없음","모델 파일 미배치","python\\model\\ 에 .pth 파일 3개 배치"],
     ["model loaded successfully 미출력","venv 비활성화 또는 라이브러리 미설치","venv 활성화 후 pip install -r 재실행"],
     ["포트 충돌 (EADDRINUSE)","이전 프로세스 잔존","작업 관리자에서 electron.exe 종료 후 재시작"]],
    [CW*0.32, CW*0.26, CW*0.42]
))
story += [sp(2), h2("8.1 전체 설치 순서 요약"), sp(1)]

steps = [
    "Git 설치  →  git --version 확인",
    "Node.js 설치  →  node --version 확인",
    "Python 설치 (Add to PATH 필수)  →  python --version 확인",
    "git clone https://github.com/RealGain-5/BE.git rcp_5th",
    "cd rcp_5th",
    "npm install",
    "cd src\\renderer  &&  npm install  &&  cd ..\\..  ",
    "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser  (최초 1회)",
    "python -m venv venv",
    ".\\venv\\Scripts\\Activate.ps1",
    "python -m pip install --upgrade pip",
    "pip install -r python\\requirements.txt",
    "python\\model\\ 에 .pth 파일 3개 배치 (7.1절 참고)",
    "npm run dev  →  model loaded successfully 확인",
]
for i, step in enumerate(steps, 1):
    story.append(Paragraph(
        f"{i:02d}.  {step}",
        ParagraphStyle(f"step{i}", fontName=BF, fontSize=9,
            backColor=C_CODE_BG if i % 2 == 0 else colors.white,
            leftIndent=8, spaceAfter=2, leading=14)
    ))
story.append(sp(2))

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=16*mm, bottomMargin=16*mm,
    title="RCPVMS 초기 세팅 가이드",
    author="Joe", subject="RCPVMS Setup Guide")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF created: {OUTPUT}")
