# SimpliSQL - DuckDB SQL Query Editor

A feature-rich, professional SQL query editor powered by **DuckDB** with AI assistance, workflow automation, and advanced data management capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-Latest-green)
![DuckDB](https://img.shields.io/badge/DuckDB-Latest-orange)

## ✨ Features

# SimpliSQL V2

SimpliSQL V2 is a desktop DuckDB workbench built with PyQt6. It focuses on local file analytics, SQL authoring, workflow-style data operations, export/audit support, and a local AI assistant for DuckDB SQL generation.

## What it does

- Run DuckDB SQL against local data
- Import and work with CSV, Excel, XML, JSON, ZIP, and Parquet
- Save query outputs directly to CSV
- Generate PDF audit summaries for `Run to Store`
- Build reusable views and workflow steps
- Use a local GGUF model to generate or refine DuckDB SQL

## Current AI model behavior

The AI assistant is local-only in the current V1 implementation.

- Uses `llama-cpp-python`
- Loads `.gguf` files from the local `models/` folder
- No Ollama dependency
- No cloud provider dependency
- Table-aware prompting with selected-table schema, aliases, and sample rows
- Table-name repair for selected uploaded files
- Prompt budget management for smaller-context local models

Important notes:

- The assistant is locked to SQL generation
- It prefers selected table names, not invented filenames

## Main features

### SQL and data work

- SQL editor and results grid
- Direct DuckDB querying over uploaded parquet-backed tables
- Cross-dialect date/time normalization for common SQL variants
- Safer physical table rewriting for uploaded parquet files
- Advanced load dialog with column selection, preview size, and optional `WHERE`

### File handling

- CSV / TXT
- Excel (`.xlsx`, `.xls`)
- XML
- JSON
- ZIP archives
- Parquet

### Exports and audit

- Export query output to CSV with `COPY`
- Run-to-store audit JSONL log with hash chaining
- PDF audit report generation

### Views and workflows

- Save reusable views
- Build workflow steps for repeated data operations
- Store local workflow metadata in `Auto_Workflow/`

## Project layout

```text
V1/
|-- Simplisql.py
|-- SimpliSQL.spec
|-- build.bat
|-- requirements.txt
|-- README.md
|-- ai/
|   |-- ai_assistant_new.py
|   `-- local_model.py
|-- core/
|   |-- query_manager.py
|   |-- query_helpers.py
|   |-- workflow_manager.py
|   |-- export_utils.py
|   |-- file_upload.py
|   `-- file_utilities.py
|-- ui/
|-- utils/
|-- docs/
|-- models/
|-- Auto_Workflow/
|-- build/
`-- dist/
```

## Requirements

- Python 3.9+
- Windows is the primary packaged target
- More RAM helps for larger parquet files and local GGUF models

Python packages are listed in [requirements.txt](c:/Users/Chandana/SimpliSql/.venv/V1/requirements.txt) and currently include:

- `duckdb`
- `pandas`
- `PyQt6`
- `pyarrow`
- `pyperclip`
- `reportlab`
- `matplotlib`
- `plotly`
- `llama-cpp-python`
- `pyinstaller`
- `pytest`
- `pytest-cov`

## Running from source

From the V1 folder:

```powershell
cd c:\Users\abcd\SimpliSql\.venv\V1
c:\Users\abcd\SimpliSql\.venv\Scripts\python.exe Simplisql.py
```

If you prefer a virtual environment shell first:

```powershell
cd c:\Users\abcd\SimpliSql\.venv\V1
..\Scripts\activate
python Simplisql.py
```

## Local AI setup

1. Place one or more `.gguf` files in:

```text
V1/models/
```

2. Launch the app.
3. Open the AI assistant.
4. Select a discovered model if needed.
5. The default model can be preloaded in the background on app start.

Notes:

- Model discovery is folder-driven; there is no hardcoded model registry.
- The app tries to use the model's native context length when possible.
- `SIMPLISQL_CONTEXT_LENGTH` can override the context size if needed.

## Building the executable

V1 already includes a PyInstaller spec and build script.

### Build with the spec

```powershell
cd c:\Users\Chandana\SimpliSql\.venv\V1
c:\Users\Chandana\SimpliSql\.venv\Scripts\python.exe -m PyInstaller SimpliSQL.spec --noconfirm
```

### Or use the batch file

```powershell
cd c:\Users\Chandana\SimpliSql\.venv\V1
.\build.bat
```

Build output:

```text
V1/dist/SimpliSQL/SimpliSQL.exe
```

## Runtime data locations

SimpliSQL stores local runtime state under:

- `Auto_Workflow/`
- `utils/ParquetFiles/` or `ParquetFiles/`
- `dist/SimpliSQL/_internal/...` when packaged

Common runtime files:

- `Auto_Workflow/ai_config.json`
- `Auto_Workflow/ai_chat_history.json`
- `Auto_Workflow/saved_queries.json`
- `Auto_Workflow/run_to_store_audit.jsonl`

## Security and privacy notes

- Local GGUF AI mode does not require cloud calls
- Uploaded data and generated exports remain local unless the user explicitly copies or shares them
- Review build/runtime artifacts before distributing the repo or packaged output

## Known current behavior

- The AI assistant is SQL-only
- The prompt builder uses selected-table schema and aliases to reduce hallucinated table names
- Follow-up generation supports cancellation, but cancellation is best-effort if a backend call is already deep in a non-streaming fallback
- Physical table rewriting is conservative and prefers leaving SQL unchanged rather than making unsafe rewrites

## Troubleshooting

### No model appears in the AI assistant

- Make sure at least one `.gguf` file exists in `V1/models/`
- Reopen the assistant or reload the model list

### Prompt too large for current model context

- Reduce selected tables in AI context
- Turn off current-query context
- Use a model with a larger context length

### Query fails on uploaded table names

- Confirm the table is uploaded and selected
- Confirm the generated SQL uses CTE alias columns after projection
- If a query references physical uploaded files, the execution path should rewrite them to `read_parquet(...)` when safe

### Packaged EXE issues

- Verify `dist/SimpliSQL/_internal/` exists next to the exe
- Verify the bundled or copied `models/` folder is present if AI is required

## Testing

Run tests from V1:

```powershell
cd c:\Users\Chandana\SimpliSql\.venv\V1
c:\Users\Chandana\SimpliSql\.venv\Scripts\python.exe -m pytest tests -v
```

## Notes for reviewers

If you are reviewing V1 for AppSec / InfoSec:

- treat it as a local desktop application, not a network service
- inspect runtime JSON and audit files separately from source
- review packaged `dist/` output independently from source checkout
