# System Context - RCPVMS

The system context diagram shows RCPVMS and its interactions with users and external systems.

**RCPVMS** is an Electron desktop application for analyzing RCP (Reactor Coolant Pump) vibration data (.BIN files) using a ResNet18 deep learning model. It classifies each of the 4 RCP orbits (RCP1A, RCP1B, RCP2A, RCP2B) as normal or abnormal, and provides GradCAM visualizations for interpretability.

```mermaid
C4Context
  title System Context - RCPVMS (RCP Vibration Management System)

  Person(analyst, "Vibration Analyst", "Inspects RCP vibration data and reviews classification results")

  System(rcpvms, "RCPVMS", "Electron desktop app that analyzes RCP vibration BIN files using ResNet18 deep learning, classifying orbits as normal/abnormal with GradCAM visualizations")

  System_Ext(filesystem, "Local Filesystem", "Stores .BIN vibration data files, exported results (JSON/CSV/Excel), and temporary GradCAM images")
  SystemDb_Ext(sqlite, "SQLite Database", "Stores user accounts and system logs in the app's userData directory")

  Rel(analyst, rcpvms, "Selects BIN files, views results, exports reports")
  Rel(rcpvms, filesystem, "Reads BIN files, writes exported results and temp images")
  Rel(rcpvms, sqlite, "Reads/writes user credentials and logs", "better-sqlite3")
```

## Key Interactions

| From | To | Description |
|------|-----|-------------|
| Analyst | RCPVMS | Selects BIN files (single or batch), views inference results with overlays, exports to JSON/CSV/Excel |
| RCPVMS | Filesystem | Reads .BIN vibration data, writes exported analysis reports and temporary GradCAM visualization images |
| RCPVMS | SQLite | Manages user accounts (registration, login) and system activity logs |
