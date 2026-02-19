const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
        BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak } = require('docx');

// ── Shared Styles ──
const tb = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const cellBorders = { top: tb, bottom: tb, left: tb, right: tb };
const headerShading = { fill: "1B3A5C", type: ShadingType.CLEAR };
const subHeaderShading = { fill: "E8EEF4", type: ShadingType.CLEAR };
const codeShading = { fill: "F5F5F5", type: ShadingType.CLEAR };
const phaseShading = { fill: "F0F6FF", type: ShadingType.CLEAR };

const hWhite = { font: "Arial", color: "FFFFFF", bold: true, size: 20 };
const hDark = { font: "Arial", color: "1B3A5C", bold: true, size: 20 };
const bodyFont = { font: "Arial", size: 20 };
const codeFont = { font: "Consolas", size: 18 };
const smallFont = { font: "Arial", size: 18 };

function headerCell(text, width) {
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: headerShading, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, ...hWhite })] })]
  });
}

function subHeaderCell(text, width) {
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: subHeaderShading, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, ...hDark })] })]
  });
}

function cell(text, width, opts = {}) {
  const runs = Array.isArray(text)
    ? text
    : [new TextRun({ text, ...(opts.code ? codeFont : bodyFont) })];
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: opts.shading || undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.LEFT,
      spacing: { before: 60, after: 60 },
      children: runs })]
  });
}

function multiLineCell(lines, width, opts = {}) {
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: opts.shading || undefined,
    verticalAlign: VerticalAlign.TOP,
    children: lines.map(line => new Paragraph({
      spacing: { before: 40, after: 40 },
      children: Array.isArray(line) ? line : [new TextRun({ text: line, ...(opts.code ? codeFont : bodyFont) })]
    }))
  });
}

function codeBlock(lines) {
  return new Table({
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: cellBorders, width: { size: 9360, type: WidthType.DXA },
        shading: codeShading,
        children: lines.map(l => new Paragraph({
          spacing: { before: 20, after: 20 },
          children: [new TextRun({ text: l, ...codeFont })]
        }))
      })]
    })]
  });
}

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    children: [new TextRun(text)] });
}
function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun(text)] });
}
function heading3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3,
    children: [new TextRun(text)] });
}
function body(texts) {
  const runs = texts.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, ...bodyFont });
    return new TextRun({ ...bodyFont, ...t });
  });
  return new Paragraph({ spacing: { before: 80, after: 80 }, children: runs });
}
function spacer() {
  return new Paragraph({ spacing: { before: 40, after: 40 }, children: [] });
}

// ── Phase Box ──
function phaseBox(number, title, description) {
  return new Table({
    columnWidths: [1200, 8160],
    rows: [new TableRow({
      children: [
        new TableCell({
          borders: cellBorders, width: { size: 1200, type: WidthType.DXA },
          shading: { fill: "1B3A5C", type: ShadingType.CLEAR },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: `STEP ${number}`, font: "Arial", color: "FFFFFF", bold: true, size: 22 })] })]
        }),
        new TableCell({
          borders: cellBorders, width: { size: 8160, type: WidthType.DXA },
          shading: phaseShading, verticalAlign: VerticalAlign.CENTER,
          children: [
            new Paragraph({ spacing: { before: 80, after: 40 },
              children: [new TextRun({ text: title, font: "Arial", bold: true, size: 22, color: "1B3A5C" })] }),
            new Paragraph({ spacing: { before: 0, after: 80 },
              children: [new TextRun({ text: description, ...smallFont, color: "444444" })] })
          ]
        })
      ]
    })]
  });
}

// ── Arrow ──
function arrow() {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: "\u25BC", font: "Arial", size: 28, color: "1B3A5C" })] });
}

// ══════════════════════ Document ══════════════════════
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 52, bold: true, color: "1B3A5C", font: "Arial" },
        paragraph: { spacing: { before: 480, after: 200 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1B3A5C", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2C5F8A", font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "3A7BBF", font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "num-overview",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "num-summary",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [
    // ── Cover Page ──
    {
      properties: {
        page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        spacer(), spacer(), spacer(), spacer(), spacer(), spacer(),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
          children: [new TextRun({ text: "RCPVMS", font: "Arial", size: 28, color: "888888" })] }),
        new Paragraph({ heading: HeadingLevel.TITLE,
          children: [new TextRun("IPC Data Flow")] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 400 },
          children: [new TextRun({ text: "Batch Processing Scenario", font: "Arial", size: 32, color: "2C5F8A", bold: true })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
          children: [new TextRun({ text: "\u2500".repeat(40), font: "Arial", size: 20, color: "CCCCCC" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "Technical Reference Document", font: "Arial", size: 22, color: "666666" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
          children: [new TextRun({ text: "Version 1.0", font: "Arial", size: 20, color: "888888" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "2026-02-19", font: "Arial", size: 20, color: "888888" })] }),
      ]
    },

    // ── Main Content ──
    {
      properties: {
        page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "RCPVMS IPC Data Flow - Batch Processing", font: "Arial", size: 16, color: "999999" })] })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", ...smallFont, color: "999999" }),
                     new TextRun({ children: [PageNumber.CURRENT], ...smallFont, color: "999999" }),
                     new TextRun({ text: " / ", ...smallFont, color: "999999" }),
                     new TextRun({ children: [PageNumber.TOTAL_PAGES], ...smallFont, color: "999999" })] })] })
      },
      children: [
        // ── 1. Overview ──
        heading1("1. Overview"),
        body(["RCPVMS(RCP Vibration Monitoring System)는 Electron 기반 데스크톱 애플리케이션으로, BIN 형식의 진동 데이터 파일을 AI 모델(ResNet18)로 분석하여 정상/비정상을 판별한다. 배치 처리(Batch Processing)는 다수의 BIN 파일을 병렬로 동시 분석하는 핵심 기능이다."]),
        spacer(),
        body([{ text: "관여 프로세스: ", bold: true }, "Renderer (React UI), Preload (Context Bridge), Main (Node.js), Python Worker (PyTorch Daemon)"]),
        spacer(),
        body([{ text: "통신 방식: ", bold: true }]),
        new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
          children: [new TextRun({ text: "Renderer \u2194 Main: Electron IPC (invoke/handle, send/on)", ...bodyFont })] }),
        new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
          children: [new TextRun({ text: "Main \u2194 Python: Child Process stdin/stdout (JSON Lines Protocol)", ...bodyFont })] }),
        spacer(),

        // ── 2. Architecture ──
        heading1("2. Process Architecture"),
        body(["배치 처리에 관여하는 4개 계층과 그 역할은 다음과 같다."]),
        spacer(),

        new Table({
          columnWidths: [2000, 2500, 4860],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("Process", 2000), headerCell("Source", 2500), headerCell("Role", 4860)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 2000), cell("ModelInference.jsx", 2500),
              cell("UI 렌더링, 사용자 액션 수신, 배치 결과 실시간 표시", 4860)
            ]}),
            new TableRow({ children: [
              cell("Preload", 2000), cell("preload/index.ts", 2500),
              cell("Context Isolation 하에서 IPC 채널을 window.api로 래핑", 4860)
            ]}),
            new TableRow({ children: [
              cell("Main", 2000), cell("main/index.ts", 2500),
              cell("IPC 핸들러 등록, PythonService 조율, 파일 시스템 접근", 4860)
            ]}),
            new TableRow({ children: [
              cell("Python Worker", 2000), cell("inference_daemon.py", 2500),
              cell("PyTorch 모델 로드, BIN 파싱, 추론, GradCAM 생성", 4860)
            ]}),
          ]
        }),
        spacer(),

        body(["Main Process 내부에서 PythonService \u2192 PythonDaemonPool \u2192 Worker 순으로 계층화되어 있으며, DaemonPool은 N개의 Python 프로세스를 Warm Pool로 관리한다."]),

        // ── 3. Step-by-step Data Flow ──
        new Paragraph({ children: [new PageBreak()] }),
        heading1("3. Step-by-Step Data Flow"),
        body(["사용자가 배치 분석을 수행하는 전체 시나리오를 9단계로 분해하여 각 단계의 데이터 흐름을 기술한다."]),
        spacer(),

        // ── STEP 1 ──
        phaseBox("1", "BIN File Selection (Multi-select)", "사용자가 다수의 BIN 파일을 선택한다."),
        spacer(),

        heading3("3.1 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.selectBinFiles()", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Preload", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Main", 1800),
              cell([new TextRun({ text: "ipcRenderer.invoke('select-bin-files')", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("OS", 1800),
              cell([new TextRun({ text: "dialog.showOpenDialog({ multiSelections })", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2190", 1200, { align: AlignmentType.CENTER }),
              cell("Renderer", 1800),
              cell([new TextRun({ text: "string[] | null", ...codeFont }), new TextRun({ text: "  (file paths array)", ...bodyFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.1.1 Renderer State Update"),
        body(["Renderer는 반환된 경로 배열을 기존 대기 목록과 병합한다. 중복 경로는 Set으로 필터링되며, 각 파일은 아래 구조의 객체로 관리된다."]),
        codeBlock([
          "{ path: string, status: 'pending', result: null, error: null }"
        ]),
        spacer(),

        arrow(),

        // ── STEP 2 ──
        phaseBox("2", "Concurrency Level Configuration", "사용자가 병렬 처리 수준(동시 실행 워커 수)을 설정한다."),
        spacer(),

        heading3("3.2 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.setConcurrencyLevel(level: number)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Preload", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Main", 1800),
              cell([new TextRun({ text: "ipcRenderer.invoke('set-concurrency-level', level)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("PythonService", 1800),
              cell([new TextRun({ text: "pythonService.setMaxConcurrent(level)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("DaemonPool", 1800),
              cell([new TextRun({ text: "pool.resize(level)", ...codeFont }), new TextRun({ text: "  Worker 증감", ...bodyFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.2.1 Pool Resize"),
        body(["DaemonPool은 요청된 수준에 따라 Worker를 동적으로 증감한다. 새 Worker 생성 시 Python 프로세스를 spawn하고 모델 로드 완료(stderr의 'model loaded successfully' 메시지)까지 대기한다."]),
        spacer(),

        arrow(),

        // ── STEP 3 ──
        phaseBox("3", "Batch Progress Listener Registration", "Renderer가 실시간 진행 상황 수신을 위한 이벤트 리스너를 등록한다."),
        spacer(),

        heading3("3.3 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.onBatchProgress(callback)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Preload", 1800), cell("(register)", 1200, { align: AlignmentType.CENTER }),
              cell("IPC Layer", 1800),
              cell([new TextRun({ text: "ipcRenderer.on('batch-inference-progress', ...)", ...codeFont })], 4560)
            ]}),
          ]
        }),
        spacer(),
        body(["이 단계에서는 데이터가 전송되지 않는다. ", { text: "'batch-inference-progress'", font: "Consolas", size: 18 }, " 채널에 대한 리스너만 등록하며, 실제 이벤트는 STEP 6에서 Main이 Push할 때 수신된다."]),
        spacer(),

        arrow(),

        // ── STEP 4 ──
        new Paragraph({ children: [new PageBreak()] }),
        phaseBox("4", "Batch Inference Request", "Renderer가 전체 파일 경로 배열을 Main으로 전송하여 배치 추론을 요청한다."),
        spacer(),

        heading3("3.4 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.runBatchInference(binPaths)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Preload", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Main", 1800),
              cell([new TextRun({ text: "ipcRenderer.invoke('model-batch-inference', binPaths)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("PythonService", 1800),
              cell([new TextRun({ text: "pythonService.runBatchInferenceParallel(binPaths, onProgress)", ...codeFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.4.1 Request Payload"),
        codeBlock([
          "// Renderer \u2192 Main",
          "binPaths: string[]",
          '// Example: ["C:/data/sample1.bin", "C:/data/sample2.bin", ...]'
        ]),
        spacer(),

        heading3("3.4.2 Batch Orchestration (Semaphore Pattern)"),
        body(["PythonService는 ", { text: "runBatchInferenceParallel()", font: "Consolas", size: 18 }, " 메서드에서 Semaphore 패턴을 사용하여 동시 실행 수를 제어한다."]),
        spacer(),
        new Paragraph({ numbering: { reference: "num-overview", level: 0 },
          children: [new TextRun({ text: "AbortController를 초기화한다 (취소 지원용).", ...bodyFont })] }),
        new Paragraph({ numbering: { reference: "num-overview", level: 0 },
          children: [new TextRun({ text: "파일 배열을 순회하며, runningJobs.size >= maxConcurrent이면 Promise.race()로 하나가 완료될 때까지 대기한다.", ...bodyFont })] }),
        new Paragraph({ numbering: { reference: "num-overview", level: 0 },
          children: [new TextRun({ text: "유휴 슬롯이 확보되면 새 작업(runInference)을 비동기로 시작하고, runningJobs Map에 등록한다.", ...bodyFont })] }),
        new Paragraph({ numbering: { reference: "num-overview", level: 0 },
          children: [new TextRun({ text: "각 작업의 시작/완료/실패 시점에 onProgress 콜백을 호출하여 Main에 진행 상황을 보고한다.", ...bodyFont })] }),
        spacer(),

        arrow(),

        // ── STEP 5 ──
        phaseBox("5", "Worker Pool Job Distribution", "DaemonPool이 개별 추론 요청을 유휴 Worker에 분배한다."),
        spacer(),

        heading3("3.5 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("DaemonPool", 1800),
              cell([new TextRun({ text: "pool.sendCommand('analyze', { bin_path })", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("DaemonPool", 1800), cell("(dispatch)", 1200, { align: AlignmentType.CENTER }),
              cell("Worker[N]", 1800),
              cell("getIdleWorker() \u2192 sendToWorker() or pendingJobs.push()", 4560)
            ]}),
            new TableRow({ children: [
              cell("DaemonPool", 1800), cell("\u2192 stdin", 1200, { align: AlignmentType.CENTER }),
              cell("Python Process", 1800),
              cell([new TextRun({ text: "JSON.stringify(command) + '\\n'", ...codeFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.5.1 Worker Dispatch Logic"),
        body(["DaemonPool은 Worker 배열을 순회하여 ", { text: "status === 'idle'", font: "Consolas", size: 18 }, "인 Worker를 찾는다."]),
        new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
          children: [new TextRun({ text: "유휴 Worker 존재: 즉시 stdin에 JSON 메시지를 기록하고 Worker 상태를 'busy'로 변경한다.", ...bodyFont })] }),
        new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
          children: [new TextRun({ text: "유휴 Worker 부재: pendingJobs 큐에 Job을 추가하고, 다른 Worker가 완료될 때 자동 배정한다.", ...bodyFont })] }),
        spacer(),

        heading3("3.5.2 stdin Payload (Main \u2192 Python)"),
        codeBlock([
          '{',
          '  "command": "analyze",',
          '  "payload": { "bin_path": "C:/data/sample1.bin" }',
          '}'
        ]),
        spacer(),

        arrow(),

        // ── STEP 6 ──
        phaseBox("6", "Python Inference Execution", "Python Worker가 BIN 파일을 파싱하고 AI 추론을 수행한다."),
        spacer(),

        heading3("3.6 Processing Pipeline"),
        new Table({
          columnWidths: [600, 2400, 6360],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("#", 600), headerCell("Operation", 2400), headerCell("Description", 6360)
            ]}),
            new TableRow({ children: [
              cell("1", 600, { align: AlignmentType.CENTER }),
              cell("BIN Parsing", 2400),
              cell("make_orbit_pils_sec9_from_bin(bin_path) \u2192 4개 RCP 채널(RCP1A, RCP1B, RCP2A, RCP2B)의 PIL Image 생성", 6360)
            ]}),
            new TableRow({ children: [
              cell("2", 600, { align: AlignmentType.CENTER }),
              cell("Model Inference", 2400),
              cell("각 RCP Image를 ResNet18 모델에 입력하여 정상/비정상 분류 및 확률 산출", 6360)
            ]}),
            new TableRow({ children: [
              cell("3", 600, { align: AlignmentType.CENTER }),
              cell("GradCAM Generation", 2400),
              cell("각 RCP에 대해 Grad-CAM heatmap 및 overlay 이미지 생성", 6360)
            ]}),
            new TableRow({ children: [
              cell("4", 600, { align: AlignmentType.CENTER }),
              cell("Base64 Encoding", 2400),
              cell("모든 이미지(orbit, heatmap, overlay)를 Base64 문자열로 인코딩", 6360)
            ]}),
            new TableRow({ children: [
              cell("5", 600, { align: AlignmentType.CENTER }),
              cell("Final Labeling", 2400),
              cell("4개 RCP 중 하나라도 abnormal이면 final_label = 'abnormal'", 6360)
            ]}),
            new TableRow({ children: [
              cell("6", 600, { align: AlignmentType.CENTER }),
              cell("JSON Response", 2400),
              cell("결과 JSON을 stdout으로 출력 후 flush", 6360)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.6.1 stdout Response (Python \u2192 Main)"),
        codeBlock([
          '{',
          '  "status": "ok",',
          '  "type": "anlysis_result",',
          '  "data": {',
          '    "final_label": "normal" | "abnormal",',
          '    "results": {',
          '      "RCP1A": {',
          '        "prediction": "normal",',
          '        "probabilities": { "normal": 0.95, "abnormal": 0.05 }',
          '      },',
          '      "RCP1B": { ... },',
          '      "RCP2A": { ... },',
          '      "RCP2B": { ... }',
          '    },',
          '    "images": {',
          '      "RCP1A": {',
          '        "orbit": "<base64>",',
          '        "heatmap": "<base64>",',
          '        "overlay": "<base64>"',
          '      },',
          '      ...  // RCP1B, RCP2A, RCP2B',
          '    }',
          '  }',
          '}'
        ]),
        spacer(),

        arrow(),

        // ── STEP 7 ──
        new Paragraph({ children: [new PageBreak()] }),
        phaseBox("7", "Result Post-processing & Progress Push", "Main이 결과를 가공하고 Renderer에 실시간 Push한다."),
        spacer(),

        heading3("3.7 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("DaemonPool", 1800), cell("stdout \u2192", 1200, { align: AlignmentType.CENTER }),
              cell("PythonService", 1800),
              cell("JSON.parse(stdout) \u2192 completeWorkerRequest()", 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("(process)", 1200, { align: AlignmentType.CENTER }),
              cell("File System", 1800),
              cell("Base64 \u2192 Buffer \u2192 tmpdir/*.png (임시 파일 저장)", 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("(process)", 1200, { align: AlignmentType.CENTER }),
              cell("DaemonPool", 1800),
              cell([new TextRun({ text: "pool.sendCommand('timeline', { bin_path })", ...codeFont }), new TextRun({ text: " (Timeline 추가 생성)", ...bodyFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("callback", 1200, { align: AlignmentType.CENTER }),
              cell("Main (IPC)", 1800),
              cell([new TextRun({ text: "onProgress(BatchProgress)", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Renderer", 1800),
              cell([new TextRun({ text: "event.sender.send('batch-inference-progress', progress)", ...codeFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.7.1 Post-processing Detail"),
        body(["PythonService는 Python으로부터 수신한 Base64 이미지를 OS 임시 디렉토리에 PNG 파일로 저장한다. 이 파일 경로가 ", { text: "visualization", font: "Consolas", size: 18 }, " 객체에 포함되어 Renderer로 전달되며, Renderer는 ", { text: "media://", font: "Consolas", size: 18 }, " 프로토콜을 통해 해당 이미지를 로드한다."]),
        spacer(),

        heading3("3.7.2 BatchProgress Payload (Main \u2192 Renderer)"),
        body(["이 데이터는 ", { text: "event.sender.send()", font: "Consolas", size: 18 }, "를 통해 Push 방식으로 전송된다 (invoke/handle 방식이 아님)."]),
        spacer(),
        codeBlock([
          '{',
          '  total: number,           // 전체 파일 수',
          '  completed: number,       // 완료된 파일 수',
          '  failed: number,          // 실패한 파일 수',
          '  current: string,         // 현재 처리 중/완료된 파일 경로',
          '  running: string[],       // 현재 실행 중인 파일 경로 배열',
          '  runningCount: number,    // 실행 중인 파일 개수',
          '  result?: InferenceResult,// 방금 완료된 파일의 추론 결과',
          '  error?: string           // 방금 실패한 파일의 에러 메시지',
          '}'
        ]),
        spacer(),

        heading3("3.7.3 Renderer State Update (Incremental)"),
        body(["Renderer는 수신된 BatchProgress에 따라 개별 파일의 상태를 점진적으로 갱신한다."]),
        spacer(),
        new Table({
          columnWidths: [2800, 2400, 4160],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("Condition", 2800), headerCell("File Status", 2400), headerCell("Action", 4160)
            ]}),
            new TableRow({ children: [
              cell("progress.running.includes(path)", 2800),
              cell([new TextRun({ text: "'running'", ...codeFont })], 2400),
              cell("해당 파일의 UI를 실행 중 상태로 변경", 4160)
            ]}),
            new TableRow({ children: [
              cell("progress.result exists", 2800),
              cell([new TextRun({ text: "'completed'", ...codeFont })], 2400),
              cell("결과를 파일 객체에 저장, UI에 즉시 반영", 4160)
            ]}),
            new TableRow({ children: [
              cell("progress.error exists", 2800),
              cell([new TextRun({ text: "'failed'", ...codeFont })], 2400),
              cell("에러 메시지를 파일 객체에 저장, 재시도 버튼 표시", 4160)
            ]}),
          ]
        }),
        spacer(),
        body(["이 Incremental Update 패턴을 통해, 배치 전체가 완료되기를 기다리지 않고 각 파일의 결과가 완료 즉시 UI에 반영된다."]),
        spacer(),

        arrow(),

        // ── STEP 8 ──
        phaseBox("8", "Batch Completion", "모든 파일의 처리가 완료되고 최종 요약이 반환된다."),
        spacer(),

        heading3("3.8 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("PythonService", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Main", 1800),
              cell([new TextRun({ text: "Map<string, { success, error? }>", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("invoke return", 1200, { align: AlignmentType.CENTER }),
              cell("Renderer", 1800),
              cell([new TextRun({ text: "{ success, summary: { total, completed, failed } }", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.offBatchProgress()", ...codeFont }), new TextRun({ text: "  (리스너 해제)", ...bodyFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.8.1 Final Response Payload"),
        body(["Main은 개별 결과를 포함하지 않고 요약 정보만 반환한다. 개별 결과는 이미 STEP 7에서 실시간으로 전달되었기 때문이다."]),
        codeBlock([
          '{',
          '  success: true,',
          '  summary: {',
          '    total: 10,',
          '    completed: 9,',
          '    failed: 1',
          '  }',
          '}'
        ]),
        spacer(),

        heading3("3.8.2 Cleanup"),
        body(["Renderer는 ", { text: "offBatchProgress()", font: "Consolas", size: 18 }, "를 호출하여 ", { text: "ipcRenderer.removeAllListeners('batch-inference-progress')", font: "Consolas", size: 18 }, "로 이벤트 리스너를 해제한다. 로그 저장 API가 호출되어 배치 결과를 DB에 기록한다."]),
        spacer(),

        arrow(),

        // ── STEP 9 ──
        phaseBox("9", "Batch Cancellation (Alternative Flow)", "사용자가 진행 중인 배치 처리를 취소한다."),
        spacer(),

        heading3("3.9 Data Flow"),
        new Table({
          columnWidths: [1800, 1200, 1800, 4560],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("From", 1800), headerCell("Direction", 1200), headerCell("To", 1800), headerCell("Data", 4560)
            ]}),
            new TableRow({ children: [
              cell("Renderer", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Preload", 1800),
              cell([new TextRun({ text: "window.api.cancelBatchInference()", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Preload", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("Main", 1800),
              cell([new TextRun({ text: "ipcRenderer.invoke('model-batch-cancel')", ...codeFont })], 4560)
            ]}),
            new TableRow({ children: [
              cell("Main", 1800), cell("\u2192", 1200, { align: AlignmentType.CENTER }),
              cell("PythonService", 1800),
              cell([new TextRun({ text: "this.abortController.abort()", ...codeFont })], 4560)
            ]}),
          ]
        }),
        spacer(),

        heading3("3.9.1 Cancellation Mechanism"),
        body(["PythonService는 ", { text: "AbortController", font: "Consolas", size: 18 }, "를 사용한다. ", { text: "abort()", font: "Consolas", size: 18 }, " 호출 시 ", { text: "signal.aborted", font: "Consolas", size: 18 }, "가 true로 변경되며, 배치 루프는 다음 반복에서 이를 확인하고 새 작업 배정을 중단한다. 이미 Python Worker에서 실행 중인 작업은 완료까지 계속 진행된다 (Worker 프로세스를 강제 종료하지 않음)."]),

        // ── 4. IPC Channel Summary ──
        new Paragraph({ children: [new PageBreak()] }),
        heading1("4. IPC Channel Reference"),
        body(["배치 처리에 사용되는 IPC 채널의 전체 목록이다."]),
        spacer(),

        new Table({
          columnWidths: [2600, 1200, 1200, 1400, 2960],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("IPC Channel", 2600), headerCell("Pattern", 1200), headerCell("Direction", 1200),
              headerCell("Preload API", 1400), headerCell("Purpose", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "select-bin-files", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("selectBinFiles", 1400), cell("다중 파일 선택 다이얼로그", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "set-concurrency-level", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("setConcurrencyLevel", 1400), cell("병렬 처리 수준 설정", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "model-batch-inference", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("runBatchInference", 1400), cell("배치 추론 실행 요청", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "batch-inference-progress", ...codeFont })], 2600),
              cell([new TextRun({ text: "send/on", ...codeFont, bold: true })], 1200),
              cell([new TextRun({ text: "M \u2192 R", bold: true, ...bodyFont })], 1200),
              cell("onBatchProgress", 1400), cell("실시간 진행률 Push", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "model-batch-cancel", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("cancelBatchInference", 1400), cell("배치 추론 취소", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "model-inference", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("runInference", 1400), cell("단일 파일 추론 (재시도)", 2960)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "db-insert-log", ...codeFont })], 2600),
              cell("invoke/handle", 1200), cell("R \u2192 M", 1200),
              cell("saveLog", 1400), cell("배치 완료 로그 기록", 2960)
            ]}),
          ]
        }),
        spacer(),
        body([{ text: "R = Renderer, M = Main. ", font: "Arial", size: 18, color: "666666" },
              { text: "batch-inference-progress", font: "Consolas", size: 18, color: "666666" },
              { text: "만 Push(send/on) 패턴이며, 나머지는 모두 Request-Response(invoke/handle) 패턴이다.", font: "Arial", size: 18, color: "666666" }]),

        // ── 5. Data Format ──
        new Paragraph({ children: [new PageBreak()] }),
        heading1("5. Key Data Structures"),

        heading2("5.1 InferenceResult (per file)"),
        body(["PythonService가 Python 응답을 가공하여 생성하는 최종 결과 객체이다."]),
        codeBlock([
          'interface InferenceResult {',
          '  bin_path: string;                    // 원본 BIN 파일 경로',
          '  model_path: string;                  // 사용된 모델 경로',
          '  final_label: "normal" | "abnormal";  // 최종 판정',
          '  results: {                           // RCP별 추론 결과',
          '    [rcp: string]: {',
          '      prediction: string;',
          '      probabilities: { normal: number; abnormal: number };',
          '    };',
          '  };',
          '  visualization: {                     // RCP별 시각화 이미지 경로',
          '    [rcp: string]: {',
          '      orbit: string;                   // 원본 궤도 이미지 경로',
          '      gradcam: {',
          '        original: string;',
          '        heatmap: string;               // Grad-CAM 히트맵 경로',
          '        overlay: string;               // Grad-CAM 오버레이 경로',
          '      };',
          '      temporal: string[];              // 타임라인 이미지 경로 배열',
          '    };',
          '  };',
          '  temp_dir: string;                    // 임시 디렉토리 경로',
          '}'
        ]),
        spacer(),

        heading2("5.2 BatchProgress (real-time push)"),
        body(["Main에서 Renderer로 Push되는 진행 상황 객체이다."]),
        codeBlock([
          'interface BatchProgress {',
          '  total: number;           // 전체 파일 수',
          '  completed: number;       // 성공 완료 수',
          '  failed: number;          // 실패 수',
          '  current: string | null;  // 현재 처리 대상 파일 경로',
          '  running?: string[];      // 동시 실행 중인 파일 경로 배열',
          '  runningCount?: number;   // 동시 실행 수',
          '  result?: any;            // 방금 완료된 결과 (InferenceResult)',
          '  error?: string;          // 방금 발생한 에러 메시지',
          '}'
        ]),
        spacer(),

        heading2("5.3 Batch File Entry (Renderer state)"),
        body(["Renderer가 내부적으로 관리하는 파일 단위 상태 객체이다."]),
        codeBlock([
          '// batchFiles: Array<BatchFileEntry>',
          '{',
          '  path: string,',
          '  status: "pending" | "running" | "completed" | "failed",',
          '  result: InferenceResult | null,',
          '  error: string | null',
          '}'
        ]),
        spacer(),

        // ── 6. Communication Protocols ──
        heading1("6. Communication Protocols"),

        heading2("6.1 Electron IPC (Renderer \u2194 Main)"),
        new Table({
          columnWidths: [2200, 7160],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("Pattern", 2200), headerCell("Description", 7160)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "invoke / handle", ...codeFont })], 2200),
              cell("Renderer가 요청을 보내고 Main이 비동기로 응답하는 Request-Response 패턴. Promise 기반이며 에러 전파를 지원한다.", 7160)
            ]}),
            new TableRow({ children: [
              cell([new TextRun({ text: "send / on", ...codeFont })], 2200),
              cell("Main이 Renderer에 단방향으로 메시지를 전송하는 Fire-and-Forget 패턴. 배치 진행률 Push에 사용된다.", 7160)
            ]}),
          ]
        }),
        spacer(),

        heading2("6.2 JSON Lines Protocol (Main \u2194 Python)"),
        new Table({
          columnWidths: [2200, 7160],
          rows: [
            new TableRow({ tableHeader: true, children: [
              headerCell("Direction", 2200), headerCell("Description", 7160)
            ]}),
            new TableRow({ children: [
              cell("Main \u2192 Python", 2200),
              cell("Worker의 stdin에 JSON 객체를 한 줄로 기록하고 개행(\\n)으로 구분한다. 각 메시지는 command와 payload를 포함한다.", 7160)
            ]}),
            new TableRow({ children: [
              cell("Python \u2192 Main", 2200),
              cell("Worker의 stdout에 JSON 응답을 한 줄로 출력하고 sys.stdout.flush()로 즉시 전송한다. DaemonPool은 stdoutBuffer에 데이터를 축적하며 개행 단위로 파싱한다.", 7160)
            ]}),
            new TableRow({ children: [
              cell("Python stderr", 2200),
              cell("로그 메시지 및 모델 로드 완료 신호('model loaded successfully')에 사용된다. 제어 흐름에 관여하지 않는다.", 7160)
            ]}),
          ]
        }),
      ]
    }
  ]
});

// ── Generate ──
const outputPath = require('path').join(__dirname, 'IPC_배치처리_데이터흐름.docx');
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document generated: ${outputPath}`);
});
