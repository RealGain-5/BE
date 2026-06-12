const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const C = {
  titleBlue: "1F4E79", h1Blue: "1F4E79", h2Blue: "2E75B6",
  headerBg: "1F4E79", subBg: "D6E4F0", altBg: "F5F9FF",
  codeBg: "F4F4F4", warnBg: "FFF3CD", warnBorder: "FFC107",
  tipBg: "E8F5E9", tipBorder: "4CAF50",
  dangerBg: "FFEBEE", dangerBorder: "F44336",
  gray: "5A5A5A", red: "C00000", redBg: "FFF0F0",
  white: "FFFFFF", black: "000000",
};
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function sp() { return new Paragraph({ children: [new TextRun("")] }); }
function h(level, text) { return new Paragraph({ heading: level, children: [new TextRun(text)] }); }
function p(text, opts = {}) { return new Paragraph({ children: [new TextRun({ text, ...opts })] }); }
function bul(text) {
  return new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun(text)] });
}
function num(text, ref) {
  return new Paragraph({ numbering: { reference: ref, level: 0 }, children: [new TextRun(text)] });
}
function code(text) {
  return new Paragraph({ style: "CodeBlock", children: [new TextRun(text || " ")] });
}
function codeLines(lines) { return lines.map(l => code(l)); }

function callout(label, lines, bgColor, labelColor) {
  const allLines = lines.map((l, i) =>
    new Paragraph({
      shading: { fill: bgColor, type: ShadingType.CLEAR },
      spacing: i === 0 ? { before: 80 } : { before: 0 },
      children: [
        ...(i === 0 ? [new TextRun({ text: label + "  ", bold: true, color: labelColor, size: 20 })] : []),
        new TextRun({ text: l, size: 20 }),
      ]
    })
  );
  return allLines;
}

function tblRow(cells, isHeader = false, firstColBold = true) {
  return new TableRow({
    tableHeader: isHeader,
    children: cells.map((cell, i) => new TableCell({
      borders: cellBorders,
      shading: isHeader
        ? { fill: C.headerBg, type: ShadingType.CLEAR }
        : (i === 0 && firstColBold ? { fill: C.subBg, type: ShadingType.CLEAR } : { fill: C.white, type: ShadingType.CLEAR }),
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        children: [new TextRun({
          text: cell, size: 20,
          bold: isHeader || (i === 0 && firstColBold),
          color: isHeader ? C.white : C.black,
        })]
      })]
    }))
  });
}

function makeTable(header, rows, widths) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: widths,
    margins: { top: 80, bottom: 80, left: 150, right: 150 },
    rows: [
      tblRow(header, true),
      ...rows.map((r, idx) => new TableRow({
        children: r.map((cell, i) => new TableCell({
          borders: cellBorders,
          shading: i === 0
            ? { fill: C.subBg, type: ShadingType.CLEAR }
            : { fill: idx % 2 === 0 ? C.white : C.altBg, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, bold: i === 0 })] })]
        }))
      }))
    ]
  });
}

// 번호 매기기 참조 목록 (섹션별로 독립된 번호 사용)
const numRefs = Array.from({ length: 20 }, (_, i) => `num-${i + 1}`);

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 52, bold: true, color: C.titleBlue, font: "Arial" },
        paragraph: { spacing: { before: 0, after: 160 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Subtitle", name: "Subtitle", basedOn: "Normal",
        run: { size: 24, color: C.gray, font: "Arial" },
        paragraph: { spacing: { before: 0, after: 320 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: C.h1Blue, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: C.h2Blue, font: "Arial" },
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
          shading: { fill: C.codeBg, type: ShadingType.CLEAR }
        }
      },
      {
        id: "Note", name: "Note", basedOn: "Normal",
        run: { size: 18, color: C.gray, italics: true },
        paragraph: { spacing: { before: 60, after: 60 }, indent: { left: 200 } }
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
      ...numRefs.map(ref => ({
        reference: ref,
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
      }))
    ]
  },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 } } },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" } },
          children: [new TextRun({ text: "RCPVMS — 초기 세팅 가이드", color: C.gray, size: 18 })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "- ", size: 18, color: C.gray }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.gray }),
            new TextRun({ text: " -", size: 18, color: C.gray })
          ]
        })]
      })
    },
    children: [

      // ── 표지 ────────────────────────────────────────────
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("RCPVMS")] }),
      new Paragraph({ style: "Subtitle", children: [new TextRun("초기 세팅 가이드 및 별도 제공 파일 목록")] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "압축 파일 기반 프로젝트 설치 · 환경 구성 완전 가이드", size: 24, bold: true, color: C.h2Blue })]
      }),
      sp(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "대상 독자: 개발 환경 초보자  |  2026년 5월", size: 20, color: C.gray })]
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ── 목차 안내 ────────────────────────────────────────
      h(HeadingLevel.HEADING_1, "문서 구성"),
      makeTable(
        ["섹션", "내용"],
        [
          ["1", "이 문서를 읽기 전에 — 기본 개념 설명"],
          ["2", "STEP 1  필수 소프트웨어 설치"],
          ["3", "STEP 2  소스 코드 배치 (압축 파일 해제)"],
          ["4", "STEP 3  Node.js 패키지 설치"],
          ["5", "STEP 4  Python 가상환경 구성"],
          ["6", "STEP 5  앱 실행 확인"],
          ["7", "별도 제공 파일 목록 및 배치 방법"],
          ["8", "자주 발생하는 오류와 해결 방법"],
        ],
        [800, 8560]
      ),
      sp(),

      // ── 1. 기본 개념 ─────────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "1. 이 문서를 읽기 전에 — 기본 개념"),

      h(HeadingLevel.HEADING_2, "1.1 소스 코드 배치란?"),
      p("이 프로젝트의 소스 코드는 압축 파일(.zip) 형태로 직접 전달됩니다. 별도의 Git 설치나 인터넷 저장소 접속 없이, 전달받은 압축 파일을 해제하고 지정 폴더에 배치하는 것만으로 코드 준비가 완료됩니다."),
      p("압축 해제 후 폴더 안의 소스 코드 구조는 모든 환경에서 동일하므로, 이 문서의 절차를 그대로 따르면 됩니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "1.2 터미널(명령 프롬프트)이란?"),
      p("터미널은 마우스 대신 키보드로 컴퓨터에 명령을 내리는 창입니다. 이 문서에서는 Windows의 PowerShell을 사용합니다."),
      ...callout("PowerShell 여는 방법", [
        "키보드에서 Windows 키 + R 을 동시에 누릅니다.",
        "실행 창에 powershell 을 입력하고 Enter 를 누릅니다.",
        "파란 창이 뜨면 성공입니다.",
      ], "E3F2FD", "1565C0"),
      sp(),

      h(HeadingLevel.HEADING_2, "1.3 명령어 입력 방법"),
      p("이 문서에서 회색 박스에 적힌 내용은 터미널에 그대로 입력하는 명령어입니다. 한 줄씩 입력하고 Enter 키를 눌러 실행합니다."),
      ...callout("주의", [
        "명령어 앞의 번호(1. 2. 3.)는 순서 표시일 뿐, 입력하지 않습니다.",
        "# 기호로 시작하는 줄은 설명(주석)이므로 입력하지 않아도 됩니다.",
      ], "FFF8E1", "F57C00"),
      sp(),

      // ── 2. 소프트웨어 설치 ────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "2. STEP 1 — 필수 소프트웨어 설치"),
      p("아래 2가지 프로그램을 순서대로 설치합니다. 이미 설치되어 있는 항목은 건너뛰어도 됩니다."),
      sp(),

      makeTable(
        ["프로그램", "용도", "확인 명령어"],
        [
          ["Node.js v20 LTS", "앱의 JavaScript 런타임 환경", "node --version"],
          ["Python 3.12", "AI 모델 실행 환경", "python --version"],
          ["Visual Studio Code", "코드 편집기 (선택 사항)", "code --version"],
        ],
        [2200, 4200, 2960]
      ),
      sp(),

      h(HeadingLevel.HEADING_2, "2.1 Node.js 설치"),
      p("① 웹 브라우저에서 https://nodejs.org 에 접속합니다."),
      p("② [LTS] 버전(왼쪽 버튼, 현재 권장 버전)을 클릭해 설치 파일을 내려받습니다."),
      p("③ 설치 파일을 실행하고 기본값으로 설치를 완료합니다."),
      p("④ PowerShell에서 아래 두 명령어로 설치를 확인합니다."),
      ...codeLines(["node --version", "npm --version"]),
      p("결과 예시: v20.18.0 / 10.8.2"),
      sp(),

      h(HeadingLevel.HEADING_2, "2.2 Python 설치"),
      p("① 웹 브라우저에서 https://www.python.org/downloads 에 접속합니다."),
      p("② [Download Python 3.12.x] 버튼을 클릭해 설치 파일을 내려받습니다."),
      p("③ 설치 파일을 실행합니다. 이때 반드시 아래 항목을 체크해야 합니다."),
      ...callout("★ 필수 체크 항목", [
        "설치 화면 맨 아래 \"Add Python 3.12 to PATH\" 체크박스를 반드시 체크하세요.",
        "이 항목을 체크하지 않으면 이후 모든 단계에서 python 명령어가 동작하지 않습니다.",
      ], "FFEBEE", "C62828"),
      p("④ [Install Now] 를 클릭하고 설치를 완료합니다."),
      p("⑤ PowerShell에서 아래 명령어로 설치를 확인합니다."),
      code("python --version"),
      p("결과 예시: Python 3.12.x"),
      sp(),

      // ── 3. 소스 코드 배치 ─────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "3. STEP 2 — 소스 코드 배치 (압축 파일 해제)"),

      h(HeadingLevel.HEADING_2, "3.1 프로젝트를 저장할 폴더 위치 결정"),
      p("소스 코드를 저장할 폴더를 먼저 정합니다. 아래 예시에서는 D:\\projects 폴더를 사용합니다. 원하는 경로로 변경해도 됩니다."),
      ...callout("예시 배치 경로", [
        "D:\\projects\\rcp_5th  → D 드라이브에 projects 폴더 안에 저장",
        "C:\\Users\\사용자이름\\Desktop\\rcp_5th  → 바탕화면에 저장",
      ], "E3F2FD", "1565C0"),
      sp(),

      h(HeadingLevel.HEADING_2, "3.2 압축 해제 및 폴더 이동"),
      p("PowerShell을 열고 아래 명령어를 순서대로 입력합니다."),
      sp(),
      p("① 저장할 상위 폴더로 이동합니다. (D 드라이브를 사용하는 경우)"),
      code("cd D:\\projects"),
      ...callout("폴더가 없다면?", [
        "mkdir D:\\projects  를 먼저 입력해 폴더를 만든 후 위 명령어를 실행하세요.",
      ], "E8F5E9", "2E7D32"),
      sp(),
      p("② 전달받은 압축 파일의 경로를 확인하고 압축을 해제합니다."),
      p("  방법 A — Windows 탐색기 사용: 압축 파일을 우클릭 → '압축 풀기' → 저장 경로로 D:\\projects 선택"),
      p("  방법 B — PowerShell 명령어 사용 (압축 파일이 D:\\downloads\\rcp_5th.zip 인 경우):"),
      code("Expand-Archive -Path D:\\downloads\\rcp_5th.zip -DestinationPath D:\\projects"),
      sp(),
      p("③ 프로젝트 폴더 안으로 이동합니다."),
      code("cd rcp_5th"),
      sp(),
      p("④ 파일이 제대로 배치되었는지 확인합니다."),
      code("ls"),
      p("아래와 같은 파일/폴더 목록이 보이면 정상입니다."),
      ...codeLines([
        "    build/   docs/   python/   resources/   src/",
        "    CLAUDE.md   package.json   tsconfig.json   ...",
      ]),
      sp(),

      // ── 4. Node.js 패키지 설치 ───────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "4. STEP 3 — Node.js 패키지 설치"),
      p("프로젝트에서 사용하는 JavaScript 라이브러리를 설치합니다. rcp_5th 폴더 안에 있는지 확인한 뒤 진행하세요."),
      sp(),

      h(HeadingLevel.HEADING_2, "4.1 현재 위치 확인"),
      code("pwd"),
      p("출력 결과가 ...\\rcp_5th 로 끝나야 합니다. 다른 경로라면 cd 명령으로 rcp_5th 폴더로 이동하세요."),
      sp(),

      h(HeadingLevel.HEADING_2, "4.2 루트 패키지 설치"),
      p("프로젝트 루트(rcp_5th 폴더)에서 아래 명령어를 실행합니다."),
      code("npm install"),
      p("수십~수백 줄의 설치 로그가 출력됩니다. 마지막에 added N packages 메시지가 나오면 성공입니다."),
      ...callout("자동으로 실행되는 과정", [
        "npm install 완료 후 postinstall 스크립트가 자동으로 실행됩니다.",
        "이 과정에서 native 모듈(better-sqlite3)이 Electron 버전에 맞게 재빌드됩니다.",
        "오류가 발생하면 8절 '자주 발생하는 오류' 항목을 참고하세요.",
      ], "E3F2FD", "1565C0"),
      sp(),

      h(HeadingLevel.HEADING_2, "4.3 렌더러 패키지 설치"),
      p("React UI 부분의 패키지를 별도로 설치합니다."),
      ...codeLines([
        "cd src\\renderer",
        "npm install",
        "cd ..\\..",
      ]),
      p("설치 완료 후 원래 위치(rcp_5th)로 돌아옵니다."),
      sp(),

      // ── 5. Python 가상환경 ────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "5. STEP 4 — Python 가상환경 구성"),

      h(HeadingLevel.HEADING_2, "5.1 가상환경이란?"),
      p("가상환경은 이 프로젝트에서만 사용하는 별도의 Python 공간입니다. 컴퓨터 전체에 영향을 주지 않고 필요한 라이브러리만 깔끔하게 설치할 수 있습니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "5.2 PowerShell 실행 정책 설정 (최초 1회)"),
      p("Windows에서 PowerShell 스크립트 실행을 허용해야 합니다. 아래 명령어를 한 번만 실행하면 됩니다."),
      code("Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"),
      p("Y 또는 A 를 입력하고 Enter 를 눌러 확인합니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "5.3 가상환경 생성"),
      p("rcp_5th 폴더에서 아래 명령어를 실행해 가상환경을 만듭니다."),
      code("python -m venv venv"),
      p("완료되면 rcp_5th\\venv\\ 폴더가 생성됩니다. 30초~1분 소요됩니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "5.4 가상환경 활성화"),
      p("가상환경을 활성화합니다. 앞으로 Python 관련 작업을 할 때마다 이 명령어를 먼저 실행해야 합니다."),
      code(".\\venv\\Scripts\\Activate.ps1"),
      p("성공하면 PowerShell 프롬프트 맨 앞에 (venv) 가 표시됩니다."),
      ...codeLines(["# 활성화 전: PS C:\\..\\rcp_5th>", "# 활성화 후: (venv) PS C:\\..\\rcp_5th>"]),
      sp(),

      h(HeadingLevel.HEADING_2, "5.5 pip 업그레이드"),
      p("패키지 관리자(pip)를 최신 버전으로 업그레이드합니다."),
      code("python -m pip install --upgrade pip"),
      sp(),

      h(HeadingLevel.HEADING_2, "5.6 Python 라이브러리 설치"),
      p("AI 모델 실행에 필요한 라이브러리를 설치합니다."),
      code("pip install -r python\\requirements.txt"),
      p("PyTorch 등 용량이 큰 라이브러리가 포함되어 있어 네트워크 속도에 따라 5~20분 소요됩니다."),
      sp(),
      ...callout("GPU 사용 시 (선택 사항)", [
        "NVIDIA GPU가 있는 경우 아래 명령어로 CUDA 버전 PyTorch를 설치하면 추론 속도가 빨라집니다.",
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "(기본 설치는 CPU 버전으로도 정상 동작합니다.)",
      ], "E8F5E9", "2E7D32"),
      sp(),

      h(HeadingLevel.HEADING_2, "5.7 설치 확인"),
      p("주요 라이브러리가 제대로 설치되었는지 확인합니다."),
      code("python -c \"import torch, numpy, PIL, scipy; print('OK')\""),
      p("OK 가 출력되면 정상입니다."),
      sp(),

      // ── 6. 앱 실행 확인 ──────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "6. STEP 5 — 앱 실행 확인"),

      h(HeadingLevel.HEADING_2, "6.1 모델 파일 배치 확인 (중요)"),
      p("앱을 실행하기 전에 AI 모델 파일이 올바른 위치에 있는지 반드시 확인해야 합니다."),
      p("아래 명령어로 확인합니다."),
      code("ls python\\model"),
      p("아래 3개 파일이 모두 보여야 합니다."),
      ...codeLines([
        "    resnet18_orbit_multiscale.pth   (42.7 MB)",
        "    orbit_cnn1d.pth                 ( 2.8 MB)",
        "    orbit_mae.pth                   (30.3 MB)",
      ]),
      p("파일이 없다면 7절을 참고해 담당자에게 파일을 요청한 후 배치하십시오."),
      sp(),

      h(HeadingLevel.HEADING_2, "6.2 개발 모드 실행"),
      p("rcp_5th 폴더에서 아래 명령어로 앱을 실행합니다."),
      code("npm run dev"),
      p("처음 실행 시 아래 메시지들이 순서대로 출력됩니다."),
      ...codeLines([
        "  > vite dev ...",
        "  > electron .",
        "  [Python] model loaded successfully    ← 이 메시지가 나오면 준비 완료",
      ]),
      ...callout("앱 창이 뜨지 않는다면?", [
        "터미널에서 ERROR 메시지를 확인하세요.",
        "모델 파일이 없는 경우 'FileNotFoundError' 오류가 납니다.",
        "Python 가상환경이 활성화(venv)되어 있는지 확인하세요.",
      ], "FFF3CD", "E65100"),
      sp(),

      h(HeadingLevel.HEADING_2, "6.3 정상 동작 확인"),
      makeTable(
        ["확인 항목", "방법", "정상 결과"],
        [
          ["앱 창 실행", "npm run dev 후 대기", "Electron 창이 열림"],
          ["Python 데몬 준비", "터미널 출력 확인", "model loaded successfully 출력"],
          ["탭 전환", "앱 상단 탭 클릭", "4개 탭이 정상 전환됨"],
          ["BIN 파일 로딩", "RCPVMS 뷰어 탭에서 파일 선택", "궤도 이미지 표시됨"],
        ],
        [2200, 2800, 4360]
      ),
      sp(),

      // ── 7. 별도 제공 파일 ─────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "7. 별도 제공 파일 목록 및 배치 방법"),
      p("아래 파일들은 소스 코드 압축 파일에 포함되지 않으므로 담당자에게 별도로 요청해야 합니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "7.1 AI 모델 체크포인트 (필수)"),
      p("AI 추론 기능을 사용하려면 아래 3개 파일이 반드시 필요합니다."),
      sp(),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3200, 1400, 1400, 3360],
        margins: { top: 80, bottom: 80, left: 150, right: 150 },
        rows: [
          tblRow(["파일명", "크기", "용도", "배치 경로"], true, false),
          ...[
            ["resnet18_orbit_multiscale.pth", "42.7 MB", "앙상블 ResNet18 모델", "python\\model\\"],
            ["orbit_cnn1d.pth", "2.8 MB", "앙상블 CNN1D 모델", "python\\model\\"],
            ["orbit_mae.pth", "30.3 MB", "MAE 이상 탐지 모델", "python\\model\\"],
          ].map((r, idx) => new TableRow({
            children: r.map((cell, i) => new TableCell({
              borders: cellBorders,
              shading: { fill: idx % 2 === 0 ? C.white : C.altBg, type: ShadingType.CLEAR },
              children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, font: i === 0 ? "Courier New" : "Arial" })] })]
            }))
          }))
        ]
      }),
      sp(),
      p("배치 방법: rcp_5th\\python\\model\\ 폴더가 없으면 아래 명령어로 생성합니다."),
      code("mkdir python\\model"),
      p("전달받은 .pth 파일을 해당 폴더에 복사합니다."),
      sp(),

      h(HeadingLevel.HEADING_2, "7.2 측정 데이터 파일 (기능 사용 시 필요)"),
      p("AI 분석 및 궤도 시각화 기능을 사용하려면 측정 데이터 파일이 필요합니다."),
      sp(),
      makeTable(
        ["파일 종류", "확장자", "용도", "배치 경로"],
        [
          ["RCPVMS 측정 데이터", ".BIN", "궤도 뷰어, 앙상블/MAE 분석", "임의 경로 (앱에서 직접 선택)"],
          ["DMD 계측 데이터", ".dmd (바이너리)", "DMD 분석 탭 — BIN 변환 소스", "임의 경로 (앱에서 직접 선택)"],
        ],
        [2400, 1200, 2760, 2800 - 200]
      ),
      ...callout("참고", [
        "BIN 파일과 DMD 파일은 특정 경로에 고정하지 않아도 됩니다.",
        "앱 실행 후 각 탭에서 '파일 선택' 버튼으로 원하는 파일을 직접 불러옵니다.",
      ], "E3F2FD", "1565C0"),
      sp(),

      h(HeadingLevel.HEADING_2, "7.3 소스 코드 압축 파일에 미포함 항목 전체 목록"),
      p("아래 항목들은 소스 코드 압축 파일에 포함되지 않습니다. 각 항목에 대한 조치 방법을 참고하십시오."),
      sp(),
      makeTable(
        ["미포함 항목", "이유", "조치"],
        [
          ["python/model/*.pth", "파일 크기 큼 (총 ~76 MB)", "7.1절 참고 — 담당자에게 요청"],
          ["data/ 및 data/*", "대용량 측정 데이터", "담당자에게 데이터 별도 수령"],
          ["dmd/ 및 dmd/*", "대용량 DMD 원본 파일", "담당자에게 데이터 별도 수령"],
          ["venv/", "Python 가상환경 (로컬 생성)", "5절 절차대로 직접 생성"],
          ["node_modules/", "Node 패키지 (로컬 설치)", "npm install 로 자동 설치"],
          ["dist/", "빌드 산출물", "npm run build:win 으로 생성"],
        ],
        [2800, 2800, 3760]
      ),
      sp(),

      // ── 8. 오류 해결 ─────────────────────────────────────
      new Paragraph({ children: [new PageBreak()] }),
      h(HeadingLevel.HEADING_1, "8. 자주 발생하는 오류와 해결 방법"),
      sp(),

      makeTable(
        ["오류 메시지", "원인", "해결 방법"],
        [
          [
            "'python' is not recognized",
            "Python PATH 미등록",
            "Python 재설치 시 'Add to PATH' 체크 필수"
          ],
          [
            "실행 정책 오류 (Activate.ps1)",
            "PowerShell 스크립트 실행 차단",
            "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 실행"
          ],
          [
            "npm install 중 node-gyp 오류",
            "네이티브 모듈 빌드 실패",
            "npx @electron/rebuild 를 별도 실행"
          ],
          [
            "FileNotFoundError: .pth 파일 없음",
            "모델 파일 미배치",
            "7.1절 참고하여 python\\model\\ 에 .pth 파일 배치"
          ],
          [
            "model loaded successfully 미출력",
            "Python 가상환경 비활성화 또는 라이브러리 미설치",
            "venv 활성화 후 pip install -r python\\requirements.txt 재실행"
          ],
          [
            "포트 충돌 오류 (EADDRINUSE)",
            "이전 프로세스가 남아있음",
            "작업 관리자에서 electron.exe, python.exe 종료 후 재시작"
          ],
        ],
        [2800, 2400, 4160]
      ),
      sp(),

      h(HeadingLevel.HEADING_2, "8.1 버전 확인 명령어 모음"),
      p("설치 완료 후 아래 명령어로 모든 버전을 한 번에 확인할 수 있습니다."),
      ...codeLines([
        "node --version",
        "npm --version",
        "python --version",
        "pip --version",
      ]),
      sp(),

      h(HeadingLevel.HEADING_2, "8.2 전체 설치 순서 요약"),
      num("Node.js 설치 → node --version 확인", "num-1"),
      num("Python 설치 (Add to PATH 필수) → python --version 확인", "num-1"),
      num("전달받은 압축 파일 해제 → 프로젝트 폴더(rcp_5th)로 이동", "num-1"),
      num("npm install", "num-1"),
      num("cd src\\renderer && npm install && cd ..\\..  ", "num-1"),
      num("Set-ExecutionPolicy RemoteSigned -Scope CurrentUser  (최초 1회)", "num-1"),
      num("python -m venv venv", "num-1"),
      num(".\\venv\\Scripts\\Activate.ps1", "num-1"),
      num("python -m pip install --upgrade pip", "num-1"),
      num("pip install -r python\\requirements.txt", "num-1"),
      num("python\\model\\ 폴더에 .pth 파일 3개 배치 (7.1절 참고)", "num-1"),
      num("npm run dev  → model loaded successfully 확인", "num-1"),
      sp(),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:\\Users\\yunha\\Desktop\\rcp_5th\\docs\\RCPVMS_초기세팅가이드.docx", buffer);
  console.log("DOCX created: RCPVMS_초기세팅가이드.docx");
});
