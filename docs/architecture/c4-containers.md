# Container Diagram - RCPVMS

The container diagram shows the major deployable/runnable units inside RCPVMS and how they communicate.

```mermaid
C4Container
  title Container Diagram - RCPVMS

  Person(analyst, "Vibration Analyst", "Inspects RCP vibration data")

  System_Boundary(rcpvms, "RCPVMS Desktop Application") {
    Container(renderer, "React SPA", "React 19, Vite, JSX", "Login form, analysis dashboard, batch processing UI, result visualization with orbit/heatmap/overlay images")
    Container(preload, "Preload Bridge", "Electron contextBridge", "Exposes secure IPC API to renderer: auth, inference, file dialogs, export")
    Container(main, "Electron Main Process", "TypeScript, Electron 38", "Orchestrates IPC handlers, manages Python daemon pool, handles file dialogs and export logic")
    Container(pythonPool, "Python Daemon Pool", "Python 3, PyTorch, ResNet18", "1-4 persistent daemon processes that load the trained model at startup and process inference commands via stdin/stdout JSON protocol")
    ContainerDb(sqlite, "SQLite Database", "better-sqlite3", "User accounts (hashed passwords) and system activity logs")
    Container(sessionStore, "Session Store", "electron-store", "Persists auth tokens and user info between sessions")
  }

  System_Ext(filesystem, "Local Filesystem", "BIN files, exported results, temp images")

  Rel(analyst, renderer, "Uses", "Electron window")
  Rel(renderer, preload, "Calls API methods", "window.api.*")
  Rel(preload, main, "Sends IPC messages", "ipcRenderer.invoke")
  Rel(main, pythonPool, "Sends analyze/timeline commands", "stdin JSON, stdout JSON")
  Rel(main, sqlite, "Reads/writes", "better-sqlite3")
  Rel(main, sessionStore, "Reads/writes auth state", "electron-store")
  Rel(main, filesystem, "Reads BIN files, writes exports and temp images", "fs / ExcelJS")
  Rel(pythonPool, filesystem, "Reads BIN files, loads model weights", "PyTorch / PIL")
```

## Container Descriptions

| Container | Technology | Responsibility |
|-----------|-----------|----------------|
| **React SPA** | React 19, Vite | Login/register form, model inference UI (single + batch), result display with orbit images, heatmaps, GradCAM overlays, export controls |
| **Preload Bridge** | Electron contextBridge | Security boundary that exposes a curated `window.api` object to the renderer, mapping method calls to IPC invocations |
| **Electron Main Process** | TypeScript, Electron | Central orchestrator: registers IPC handlers for auth, file selection, inference, batch processing, concurrency control, and result export (JSON/CSV/Excel with images) |
| **Python Daemon Pool** | Python, PyTorch, ResNet18 | 1-4 persistent child processes that pre-load the trained ResNet18 model at startup (eliminating cold-start latency), receive commands via stdin JSON, and return results via stdout JSON. Supports parallel batch inference with configurable concurrency |
| **SQLite Database** | better-sqlite3 | Stores `users` table (username, hashed password) and `system_logs` table (action, details, timestamp) |
| **Session Store** | electron-store | File-based key-value store for persisting auth tokens and user info across app restarts |
