const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };
const headerShading = { fill: "1F4E79", type: ShadingType.CLEAR };
const subHeaderShading = { fill: "D6E4F0", type: ShadingType.CLEAR };
const altRowShading = { fill: "F5F9FF", type: ShadingType.CLEAR };

function h(level, text) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}

function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })] });
}

function code(text) {
  const lines = text.split('\n');
  return lines.map(line =>
    new Paragraph({
      style: "CodeBlock",
      children: [new TextRun({ text: line === '' ? ' ' : line })]
    })
  );
}

function bullet(text, ref = "bullet-list") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    children: [new TextRun(text)]
  });
}

function tableRow(cells, isHeader = false) {
  return new TableRow({
    tableHeader: isHeader,
    children: cells.map((cell, i) => new TableCell({
      borders: cellBorders,
      shading: isHeader ? headerShading : (i === 0 ? subHeaderShading : { fill: "FFFFFF", type: ShadingType.CLEAR }),
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        children: [new TextRun({
          text: cell,
          bold: isHeader,
          color: isHeader ? "FFFFFF" : "000000",
          size: 20
        })]
      })]
    }))
  });
}

function table2col(rows, headerRow) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3120, 6240],
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    rows: [
      tableRow(headerRow, true),
      ...rows.map((r, idx) => new TableRow({
        children: [
          new TableCell({
            borders: cellBorders,
            shading: subHeaderShading,
            children: [new Paragraph({ children: [new TextRun({ text: r[0], bold: true, size: 20 })] })]
          }),
          new TableCell({
            borders: cellBorders,
            shading: idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading,
            children: [new Paragraph({ children: [new TextRun({ text: r[1], size: 20 })] })]
          })
        ]
      }))
    ]
  });
}

function table3col(rows, headerRow) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2600, 4160, 2600],
    margins: { top: 80, bottom: 80, left: 160, right: 160 },
    rows: [
      tableRow(headerRow, true),
      ...rows.map((r, idx) => new TableRow({
        children: r.map((cell, i) => new TableCell({
          borders: cellBorders,
          shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
          children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, size: 20 })] })]
        }))
      }))
    ]
  });
}

function spacer() {
  return new Paragraph({ children: [new TextRun("")] });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 52, bold: true, color: "1F4E79", font: "Arial" },
        paragraph: { spacing: { before: 0, after: 200 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Subtitle", name: "Subtitle", basedOn: "Normal",
        run: { size: 26, color: "5A5A5A", font: "Arial" },
        paragraph: { spacing: { before: 0, after: 400 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F4E79", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E75B6", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "404040", font: "Arial" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      },
      {
        id: "CodeBlock", name: "Code Block", basedOn: "Normal",
        run: { font: "Courier New", size: 18, color: "2E2E2E" },
        paragraph: {
          spacing: { before: 0, after: 0 },
          indent: { left: 360 },
          shading: { fill: "F4F4F4", type: ShadingType.CLEAR }
        }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list-2",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list-3",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list-4",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list-5",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      },
      {
        reference: "num-list-6",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 } }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" } },
          children: [new TextRun({ text: "RCPVMS 프로젝트 인수인계 문서", color: "5A5A5A", size: 18 })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "- ", size: 18, color: "5A5A5A" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "5A5A5A" }),
            new TextRun({ text: " -", size: 18, color: "5A5A5A" })
          ]
        })]
      })
    },
    children: [
      // ── 표지 ──
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("RCPVMS")] }),
      new Paragraph({ style: "Subtitle", children: [new TextRun("Reactor Coolant Pump Vibration Monitoring System")] }),
      new Paragraph({ style: "Subtitle", children: [new TextRun("프로젝트 인수인계 문서")] }),
      spacer(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "작성일: 2026년 5월", size: 20, color: "888888" })]
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ── 1. 프로젝트 개요 ──
      h(HeadingLevel.HEADING_1, "1. 프로젝트 개요 및 목적"),
      h(HeadingLevel.HEADING_2, "1.1 프로젝트 정의"),
      spacer(),
      table2col([
        ["시스템 명칭", "RCPVMS (Reactor Coolant Pump Vibration Monitoring System)"],
        ["형태", "Electron 기반 Windows 데스크톱 애플리케이션"],
        ["목적", "원자력 발전소 냉각재 펌프(RCP)의 진동 신호를 시각화하고 AI 모델로 이상 여부 자동 탐지"],
        ["대상 데이터", "RCPVMS 전용 .BIN 바이너리 파일, DMD 계측 장비 출력 파일"]
      ], ["항목", "내용"]),
      spacer(),

      h(HeadingLevel.HEADING_2, "1.2 핵심 기능 목록"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2340, 7020],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["탭", "기능"], true),
          ...[
            ["앙상블 분석", "ResNet18 + CNN1D 앙상블로 정상/이상 4-class 분류 + OOD 탐지"],
            ["MAE 분석", "Masked Autoencoder 기반 재구성 오차 이상 탐지"],
            ["DMD 분석", "DMD 파일 → RCPVMS BIN 변환 + 궤도 뷰어"],
            ["RCPVMS 뷰어", "BIN 파일 궤도(orbit) 그리드 시각화 (단일/배치)"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
              children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, size: 20 })] })]
            }))
          }))
        ]
      }),
      spacer(),

      h(HeadingLevel.HEADING_2, "1.3 모니터링 대상"),
      bullet("RCP 4개: RCP1A / RCP1B / RCP2A / RCP2B (NIMS 표준 명칭)"),
      bullet("채널 타입: 변위(orbit, ch_type=1), 가속도(ch_type=0), Keyphasor(ch_type=2)"),
      spacer(),

      // ── 2. 기술 스택 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "2. 기술 스택"),
      h(HeadingLevel.HEADING_2, "2.1 Frontend / Desktop"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 3000, 3760],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["항목", "버전", "내용"], true),
          ...[
            ["Electron", "^38.1.2", "메인 프로세스 + 브라우저 창 관리"],
            ["electron-vite", "^4.0.1", "빌드/번들링/HMR"],
            ["React", "JSX", "렌더러 UI"],
            ["TypeScript", "^5.9.2", "Main/Preload 계층"],
            ["better-sqlite3", "^12.5.0", "로컬 DB (로그, 인증)"],
            ["ExcelJS", "^4.4.0", "결과 Excel 내보내기"],
            ["electron-store", "^6.0.1", "앱 설정 영속화"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
              children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, size: 20 })] })]
            }))
          }))
        ]
      }),
      spacer(),

      h(HeadingLevel.HEADING_2, "2.2 Backend (Python 데몬)"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 3000, 3760],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["항목", "버전", "내용"], true),
          ...[
            ["Python", "3.10+ (3.12 권장)", "가상환경: venv/"],
            ["PyTorch", ">=2.0.0", "CPU; GPU는 CUDA 버전으로 교체"],
            ["torchvision", ">=0.15.0", "이미지 변환"],
            ["NumPy", ">=1.24.0", "수치 연산"],
            ["Pillow", ">=10.0.0", "이미지 생성/처리"],
            ["SciPy", ">=1.10.0", "신호 처리, 필터링"],
            ["Matplotlib", ">=3.7.0", "축 레이블 렌더링"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
              children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, size: 20 })] })]
            }))
          }))
        ]
      }),
      spacer(),

      h(HeadingLevel.HEADING_2, "2.3 IPC 통신"),
      bullet("Electron ipcMain / ipcRenderer + contextBridge"),
      bullet("Python 프로세스와 stdin/stdout JSON 스트리밍 방식으로 통신"),
      spacer(),

      // ── 3. 데이터 흐름 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "3. 데이터 흐름 (Data Flow)"),

      h(HeadingLevel.HEADING_2, "3.1 전체 아키텍처 흐름"),
      ...code(
`Renderer (React UI)
  window.api.xxx() 호출
      ↓ ipcRenderer.invoke()
Preload (contextBridge)
  window.api 객체 노출 (보안 경계)
      ↓ IPC channel
Main Process (index.ts + services)
  ipcMain.handle() → pythonService.runXxx()
  + SQLite DB 관리 (로그/인증)
      ↓ PythonDaemonPool
  Worker 1 (stdin/stdout) ... Worker N (stdin/stdout)
      ↓
inference_daemon.py (Python)
  명령 수신 → 해당 모듈 호출 → JSON 응답
  캐시: _rcpvms_header_cache / _rcpvms_orbit_cache
      ↓
  rcpvms_parser  preprocess.py  model_loader.py
  dmd_parser     infer_resnet   model_mae.py
  dmd_to_rcpvms  evaluate_ens`),
      spacer(),

      h(HeadingLevel.HEADING_2, "3.2 AI 추론 데이터 흐름"),
      ...code(
`BIN 파일
  └─► rcpvms_parser.py (헤더 파싱 + 채널 데이터 추출)
        └─► preprocess.py
              ├─ extract_rcp_xy_from_bin()  ← X/Y 변위 채널 추출
              ├─ estimate_1x_freq()         ← 1X 회전 주파수 탐지
              └─ _make_display_pil()        ← 궤도 이미지 생성
                    ↓ [filter_mode: raw / 1x / 2x / broadband / overlay]
        infer_resnet_None.py  +  train_1d_cnn.py
        (ResNet18 멀티스케일)    (OrbitCNN1D)
              ↓ probabilities        ↓ probabilities
              └──────── 앙상블 가중합 ──────────┘
                    ↓
              OOD 판정 (TV Distance + max_conf)
                    ↓
              최종 판정: normal / abnormal_typeA~C / unknown_abnormal
                    ↓
              GradCAM + Integrated Gradients 시각화`),
      spacer(),

      h(HeadingLevel.HEADING_2, "3.3 MAE 이상 탐지 흐름"),
      ...code(
`BIN 파일
  └─► Stage 1: 슬라이딩 윈도우 배치 스윕
        → 재구성 오차 점수 계산 (1D + Spectral)
        → 최고 점수 윈도우 선정
  └─► Stage 2: 선정 윈도우 재평가 (n_eval=10)
        → OR 로직: (score_1d > threshold_1d) OR (score_spec > threshold_spec)
        → 최종 판정: normal / abnormal`),
      spacer(),

      h(HeadingLevel.HEADING_2, "3.4 궤도 뷰어 데이터 흐름"),
      ...code(
`BIN 파일 선택
  └─► rcpvms-info   → 파일 메타 (채널 수, 샘플링 주파수, 총 시간)
  └─► rcpvms-orbit  → 모든 RCP/시간 윈도우 그리드 정보 계산
  └─► rcpvms-orbit-multi (배치 API)
        ↳ IntersectionObserver: 뷰포트 진입 시만 요청
        ↳ 썸네일 96px (PIL _draw_crosshair, matplotlib 생략)
        ↳ 모달 256px (render_with_axes — 축 레이블 포함)`),
      spacer(),

      // ── 4. 컴포넌트 맵 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "4. 기능 개요도 (Component Map)"),
      ...code(
`App.jsx (탭 라우터)
  ├── [앙상블 분석] ModelInference.jsx
  │     ├── AnalysisModeLayout.jsx (공통 레이아웃)
  │     │     ├── SingleFileMode.jsx
  │     │     ├── BatchFileList.jsx
  │     │     ├── BatchProgressBar.jsx
  │     │     ├── BatchActionButtons.jsx
  │     │     ├── BatchResultList.jsx
  │     │     ├── ConcurrencySelector.jsx
  │     │     └── ErrorDisplay.jsx
  │     └── labelStrategies.jsx (ensemble 판정 렌더링)
  │
  ├── [MAE 분석] MAEAnalysis.jsx
  │     ├── AnalysisModeLayout.jsx (동일 공유)
  │     └── labelStrategies.jsx (mae 판정 렌더링)
  │
  ├── [DMD 분석] DmdOrbitViewer.jsx
  │     ├── SubTabNav.jsx
  │     ├── [서브탭1] DMD 변환: FileOperationFlow.jsx
  │     └── [서브탭2] BIN 궤도 뷰어: FileOperationFlow.jsx → OrbitGrid.jsx
  │
  └── [RCPVMS 뷰어] RcpvmsOrbitViewer.jsx
        ├── FileOperationFlow.jsx
        ├── OrbitControls.jsx (필터 모드 / window_sec / axis_lim)
        └── OrbitGrid.jsx (IntersectionObserver + 배치 API)`),
      spacer(),

      // ── 5. 환경 세팅 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "5. 프로그램 코드 다운로드 및 환경 세팅"),

      h(HeadingLevel.HEADING_2, "5.1 사전 준비 (설치 필요 소프트웨어)"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 2600, 4160],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["소프트웨어", "권장 버전", "비고"], true),
          ...[
            ["Git", "최신", "https://git-scm.com"],
            ["Node.js", "20.x LTS 이상", "https://nodejs.org"],
            ["Python", "3.10 이상 (3.12 권장)", "https://python.org — \"Add to PATH\" 필수 체크"],
            ["Visual Studio Code", "최신", "선택 사항"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
              children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, size: 20 })] })]
            }))
          }))
        ]
      }),
      spacer(),

      h(HeadingLevel.HEADING_2, "5.2 코드 다운로드"),
      ...code(
`# 원하는 디렉토리로 이동 (예: D:\\projects)
cd D:\\projects

# Git 클론
git clone https://github.com/RealGain-5/BE.git rcp_5th

cd rcp_5th`),
      spacer(),

      h(HeadingLevel.HEADING_2, "5.3 Node.js 의존성 설치"),
      ...code(
`# 루트 패키지 설치
npm install

# 렌더러(React) 의존성 설치
cd src/renderer
npm install
cd ../..`),
      spacer(),
      p("※ npm install 후 postinstall 스크립트가 better-sqlite3를 Electron 버전에 맞게 자동 재빌드합니다.", { italics: true, size: 20, color: "5A5A5A" }),
      p("  오류 발생 시 수동 실행: npx @electron/rebuild", { font: "Courier New", size: 18, color: "444444" }),
      spacer(),

      h(HeadingLevel.HEADING_2, "5.4 Python 가상환경 세팅"),
      ...code(
`# 프로젝트 루트에서 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows PowerShell)
.\\venv\\Scripts\\Activate.ps1

# PowerShell 실행 정책 오류 발생 시
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# pip 최신화 및 의존성 설치
python -m pip install --upgrade pip
pip install -r python/requirements.txt`),
      spacer(),
      p("※ GPU 사용 시: requirements.txt의 torch 줄을 CUDA 버전으로 교체", { italics: true, size: 20, color: "5A5A5A" }),
      p("  예) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", { font: "Courier New", size: 18, color: "444444" }),
      spacer(),

      h(HeadingLevel.HEADING_2, "5.5 AI 모델 파일 배치"),
      p("모델 체크포인트는 Git에 포함되지 않을 수 있습니다. 아래 경로 확인 후 없으면 담당자에게 요청하십시오."),
      ...code(
`python/model/
  ├── resnet18_orbit_multiscale.pth   ← 앙상블 ResNet18
  ├── orbit_cnn1d.pth                 ← 앙상블 CNN1D
  └── orbit_mae.pth                   ← MAE 이상 탐지`),
      spacer(),

      h(HeadingLevel.HEADING_2, "5.6 설정 파일 확인"),
      ...code(
`python/
  ├── ensemble_config.json    ← 앙상블 가중치 / OOD 임계값
  └── mae_config.json         ← MAE 임계값 설정`),
      spacer(),
      p("ensemble_config.json 기본값:"),
      ...code(
`{
  "resnet_weight": 0.5,
  "cnn1d_weight": 0.5,
  "ood_threshold": 0.70,
  "tv_threshold": 0.30
}`),
      spacer(),

      // ── 6. Dev 모드 실행 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "6. 개발 모드 실행 방법"),

      h(HeadingLevel.HEADING_2, "6.1 개발 모드 실행 (HMR 포함)"),
      ...code(`npm run dev`),
      spacer(),
      p("내부 동작 순서:"),
      new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Vite 개발 서버(렌더러) 시작")] }),
      new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Electron 메인 프로세스 시작")] }),
      new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("BrowserWindow 생성 → 렌더러 URL 로드")] }),
      new Paragraph({ numbering: { reference: "num-list", level: 0 }, children: [new TextRun("Python 데몬 자동 시작: venv/Scripts/python python/inference_daemon.py")] }),
      spacer(),

      h(HeadingLevel.HEADING_2, "6.2 개발 모드 확인 포인트"),
      spacer(),
      table2col([
        ["Python 데몬 준비 확인", "콘솔에 'model loaded successfully' 출력 대기"],
        ["DevTools 열기", "F12 키 (개발 모드에서만 활성)"],
        ["메인 프로세스 로그", "Electron 터미널 창 확인"],
        ["렌더러 로그", "DevTools Console 탭 확인"]
      ], ["확인 사항", "방법"]),
      spacer(),

      h(HeadingLevel.HEADING_2, "6.3 TypeScript 타입 체크"),
      ...code(`npm run typecheck`),
      spacer(),

      // ── 7. 프로덕션 빌드 ──
      h(HeadingLevel.HEADING_1, "7. 프로덕션 빌드 방법"),
      ...code(
`# Python 데몬 PyInstaller 빌드 + Electron 앱 패키징 (Windows NSIS 설치 파일)
npm run build:win

# 압축 없이 폴더 형태로 언팩 빌드 (테스트용)
npm run build:unpack`),
      spacer(),
      p("빌드 결과물: dist/ 폴더에 .exe 설치 파일 생성"),
      spacer(),

      // ── 8. 파일 위치 참조 ──
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "8. 주요 파일 위치 빠른 참조"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3120, 6240],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["목적", "파일 경로"], true),
          ...[
            ["탭 추가/수정", "src/renderer/src/utils/tabRegistry.js"],
            ["IPC 핸들러 추가", "src/main/index.ts"],
            ["Python 서비스 메서드 추가", "src/main/services/pythonService.ts"],
            ["preload API 노출", "src/preload/index.ts"],
            ["Python 명령 처리 로직", "python/inference_daemon.py"],
            ["신호 처리 / 필터링", "python/preprocess.py"],
            ["AI 모델 정의", "python/model_mae.py, python/infer_resnet_None.py"],
            ["앙상블 설정", "python/ensemble_config.json"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading),
              children: [new Paragraph({ children: [new TextRun({ text: cell, bold: i === 0, font: i === 1 ? "Courier New" : "Arial", size: 20 })] })]
            }))
          }))
        ]
      }),
      spacer(),

      // ── 9. 미완료 작업 ──
      h(HeadingLevel.HEADING_1, "9. 미완료 작업 및 인수인계 체크리스트"),
      spacer(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3600, 1800, 3960],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          tableRow(["항목", "상태", "비고"], true),
          ...[
            ["앙상블 모델 재학습", "미완료", "OE + 복합 지표 코드 완성, 실제 재학습 필요"],
            ["일반화 평가", "미완료", "다른 날짜/조건 BIN 파일 테스트 필요"],
            ["DMD 실제 파일 변환 검증", "미완료", "5.1GB DMD 파일 변환 테스트 필요"],
            ["변환 BIN → 학습 파이프라인 투입", "미완료", "변환된 BIN 파일로 CNN 재학습 필요"]
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: i === 1
                ? { fill: "FFF0F0", type: ShadingType.CLEAR }
                : (i === 0 ? subHeaderShading : (idx % 2 === 0 ? { fill: "FFFFFF", type: ShadingType.CLEAR } : altRowShading)),
              children: [new Paragraph({
                children: [new TextRun({
                  text: cell,
                  bold: i === 0,
                  color: i === 1 ? "C00000" : "000000",
                  size: 20
                })]
              })]
            }))
          }))
        ]
      }),
      spacer(),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:\\Users\\yunha\\Desktop\\rcp_5th\\docs\\RCPVMS_인수인계.docx", buffer);
  console.log("DOCX created successfully.");
});
