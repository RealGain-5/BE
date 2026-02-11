# Component Diagram - Electron Main Process

The component diagram shows the internal structure of the Electron Main Process and how its components interact.

```mermaid
C4Component
  title Component Diagram - Electron Main Process

  Container(renderer, "React SPA", "React 19", "Frontend UI")
  Container(preload, "Preload Bridge", "contextBridge", "IPC API")
  ContainerDb(sqlite, "SQLite DB", "better-sqlite3", "Users & logs")
  Container(sessionStore, "Session Store", "electron-store", "Auth state")
  System_Ext(filesystem, "Local Filesystem", "BIN files & exports")

  Container_Boundary(main, "Electron Main Process") {
    Component(ipcHandlers, "IPC Handlers", "ipcMain.handle", "Routes 14 IPC channels: auth, file dialogs, inference, batch processing, export")
    Component(mediaProtocol, "Media Protocol", "protocol.handle", "Custom media:// protocol that resolves local image file paths for the renderer")
    Component(authService, "Auth Service", "auth.js", "Login, register, logout, session check using bcrypt password hashing")
    Component(dbModule, "Database Module", "db.js", "Schema init, log insert/query, user CRUD operations")
    Component(pythonService, "PythonService", "pythonService.ts", "Singleton orchestrator for inference: single file, sequential batch, parallel batch with progress callbacks")
    Component(daemonPool, "PythonDaemonPool", "PythonDaemonPool.ts", "Manages 1-4 Python child processes with job queue, worker lifecycle, and dynamic pool resizing")
    Component(exportLogic, "Export Logic", "ipcMain handlers", "Generates JSON, CSV (with BOM), and Excel (with embedded GradCAM images) exports via save dialogs")
  }

  Rel(preload, ipcHandlers, "IPC invoke/on", "14 channels")
  Rel(ipcHandlers, authService, "Login, register, logout, check")
  Rel(ipcHandlers, pythonService, "runInference, runBatchInferenceParallel, cancel, setConcurrency")
  Rel(ipcHandlers, exportLogic, "Export JSON/CSV/Excel")
  Rel(authService, dbModule, "findUserByUsername, createUser")
  Rel(authService, sessionStore, "Store/clear auth tokens")
  Rel(dbModule, sqlite, "SQL queries")
  Rel(pythonService, daemonPool, "sendCommand(analyze/timeline)")
  Rel(daemonPool, filesystem, "Spawns Python processes, reads BIN files")
  Rel(pythonService, filesystem, "Saves Base64 images to temp dirs")
  Rel(exportLogic, filesystem, "Writes JSON/CSV/Excel files")
  Rel(mediaProtocol, filesystem, "Resolves media:// URLs to local files")
  Rel(ipcHandlers, dbModule, "insertLog, getLogs")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Component Descriptions

| Component | File | Responsibility |
|-----------|------|----------------|
| **IPC Handlers** | `index.ts` | Registers 14 `ipcMain.handle` channels that bridge renderer requests to backend services |
| **Media Protocol** | `index.ts` | Custom `media://` protocol handler that translates image URLs to local file paths, handling Windows drive letter edge cases |
| **Auth Service** | `services/auth.js` | User authentication with bcrypt password hashing, session management via electron-store |
| **Database Module** | `database/db.js` | SQLite schema initialization, CRUD for `users` and `system_logs` tables |
| **PythonService** | `services/pythonService.ts` | Singleton that orchestrates all inference operations: single-file, sequential batch, and parallel batch with semaphore-based concurrency control, progress streaming, and cancellation support |
| **PythonDaemonPool** | `utils/PythonDaemonPool.ts` | Manages a pool of persistent Python child processes (1-4), with stdin/stdout JSON communication, job queuing, worker health monitoring, and dynamic pool resizing |
| **Export Logic** | `index.ts` (inline) | Generates three export formats: raw JSON, CSV with BOM for Excel compatibility, and XLSX with embedded GradCAM overlay images using ExcelJS |
