# Dynamic Diagram - Single File Inference Flow

Shows the step-by-step request flow when an analyst selects a BIN file and runs inference.

```mermaid
C4Dynamic
  title Dynamic Diagram - Single File Inference Flow

  Person(analyst, "Vibration Analyst", "Selects BIN file")

  Container(renderer, "React SPA", "React 19", "ModelInference component")
  Container(preload, "Preload Bridge", "contextBridge", "window.api")

  Container_Boundary(main, "Electron Main Process") {
    Component(ipc, "IPC Handlers", "ipcMain", "Route requests")
    Component(pyService, "PythonService", "Singleton", "Inference orchestrator")
    Component(pool, "PythonDaemonPool", "Pool", "Worker management")
  }

  Container(daemon, "Python Daemon", "PyTorch", "ResNet18 model loaded in memory")
  System_Ext(fs, "Filesystem", "BIN files & temp images")

  Rel(analyst, renderer, "1. Click 'Select BIN File'")
  Rel(renderer, preload, "2. window.api.selectBinFile()")
  Rel(preload, ipc, "3. ipcMain 'select-bin-file'")
  Rel(ipc, fs, "4. dialog.showOpenDialog() returns path")
  Rel(renderer, preload, "5. window.api.runInference(binPath)")
  Rel(preload, ipc, "6. ipcMain 'model-inference'")
  Rel(ipc, pyService, "7. pythonService.runInference()")
  Rel(pyService, pool, "8. pool.sendCommand('analyze', {bin_path})")
  Rel(pool, daemon, "9. Write JSON to stdin", "stdin")
  Rel(daemon, fs, "10. Read BIN, generate orbit images")
  Rel(daemon, pool, "11. Return predictions + Base64 images", "stdout JSON")
  Rel(pyService, fs, "12. Save Base64 images to temp dir")
  Rel(pyService, pool, "13. pool.sendCommand('timeline')")
  Rel(pool, daemon, "14. Generate temporal orbit images", "stdin/stdout")
  Rel(pyService, ipc, "15. Return InferenceResult")
  Rel(ipc, renderer, "16. Display results: labels, probabilities, orbit/heatmap/overlay images")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Flow Description

| Step | Action | Detail |
|------|--------|--------|
| 1-4 | **File Selection** | User clicks select button, Electron shows native file dialog, returns `.BIN` file path |
| 5-8 | **Inference Request** | Renderer calls `runInference` via IPC, PythonService routes to DaemonPool |
| 9-11 | **Python Processing** | Daemon reads BIN, generates 4 orbit images (RCP1A/1B/2A/2B), runs ResNet18 prediction + GradCAM, returns JSON with Base64-encoded images |
| 12 | **Image Persistence** | PythonService decodes Base64 images to temp files so renderer can display them via `media://` protocol |
| 13-14 | **Timeline Generation** | Second daemon command generates temporal orbit images for animation |
| 15-16 | **Result Display** | Final `InferenceResult` returned to renderer: final_label (normal/abnormal), per-RCP predictions with probabilities, visualization paths |

---

# Dynamic Diagram - Batch Parallel Inference Flow

Shows how batch inference processes multiple BIN files in parallel using the daemon pool.

```mermaid
C4Dynamic
  title Dynamic Diagram - Batch Parallel Inference

  Container(renderer, "React SPA", "React 19", "Batch UI with progress bar")

  Container_Boundary(main, "Electron Main Process") {
    Component(ipc, "IPC Handlers", "ipcMain", "Batch channel")
    Component(pyService, "PythonService", "Singleton", "Semaphore-based concurrency")
    Component(pool, "PythonDaemonPool", "Pool", "N idle workers")
  }

  Container(worker1, "Python Worker 1", "PyTorch", "ResNet18")
  Container(worker2, "Python Worker 2", "PyTorch", "ResNet18")

  Rel(renderer, ipc, "1. runBatchInference([file1..fileN])")
  Rel(ipc, pyService, "2. runBatchInferenceParallel()")
  Rel(pyService, pool, "3. sendCommand(file1) to idle worker")
  Rel(pool, worker1, "4a. Analyze file1", "stdin")
  Rel(pyService, pool, "4b. sendCommand(file2) to idle worker")
  Rel(pool, worker2, "4c. Analyze file2", "stdin")
  Rel(worker1, pyService, "5. Result for file1", "stdout")
  Rel(pyService, ipc, "6. batch-inference-progress event with result", "IPC send")
  Rel(ipc, renderer, "7. Real-time progress update + incremental result")
  Rel(worker2, pyService, "8. Result for file2", "stdout")
  Rel(pyService, pool, "9. sendCommand(file3) to freed worker")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Parallel Processing Details

- **Concurrency Control**: Semaphore pattern limits concurrent jobs to `maxConcurrent` (1-4, configurable at runtime)
- **Pool Resizing**: `setConcurrencyLevel()` dynamically adjusts the number of Python daemon workers
- **Progress Streaming**: Each completed file triggers an IPC `send` event with incremental results (not just counts)
- **Cancellation**: `AbortController` allows mid-batch cancellation; in-flight jobs complete but no new jobs start
- **Memory Optimization**: Final batch result returns only success/failure summary; full results are streamed incrementally
