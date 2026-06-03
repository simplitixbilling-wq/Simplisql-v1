# Run to Store Audit Clearance Note (V1)

## 1) Is this fully foolproof?
No software audit control is 100% foolproof.
This implementation is tamper-evident and traceable, which is the correct goal for auditability.

## 2) What controls are implemented
- Unique run identifier for every export (Run ID).
- Export data integrity hash (CSV SHA256).
- Query integrity hash (original and executed query SHA256).
- Actor and endpoint context (user, machine, IP, platform, app version).
- Append-only JSONL audit log with hash chain:
  - each record stores prev_record_hash
  - each record stores record_hash calculated from canonicalized record content
- Event coverage for success, cancellation, failure, and PDF status.
- Sidecar link file per export that ties CSV, PDF report, and audit record hash.

## 3) How to explain this to auditors
Use this statement:

"Run to Store in V1 produces a tamper-evident audit trail with end-to-end traceability.
Every execution is assigned a Run ID and produces integrity hashes for both query text and output CSV.
Audit records are written to an append-only JSONL log with cryptographic hash chaining
(prev_record_hash -> record_hash), enabling tamper detection.
Each export also generates a sidecar link document connecting the CSV artifact,
its PDF report, and the corresponding audit record hash.
Cancellation and failure events are logged as first-class audit events, not silently dropped."

## 4) Audit evidence package to provide
For one sample run, provide:
- Exported CSV file.
- PDF audit report.
- Sidecar link file (same base name as CSV, suffix _audit_link.json).
- Global run log file under V1 Auto_Workflow directory.
- Screenshot of success message with Run ID and Audit Record Hash.

## 5) Verification procedure (auditor walkthrough)
1. Match Run ID across success dialog, PDF report, sidecar link file, and JSONL log.
2. Confirm CSV SHA256 in PDF/sidecar equals hash computed on exported CSV.
3. Confirm audit_record_hash in sidecar exists as record_hash in JSONL.
4. Confirm JSONL chain continuity:
   - record[N].prev_record_hash equals record[N-1].record_hash.
5. Confirm event chronology exists for the run:
   - run_to_store success event
   - run_to_store_pdf event (success or failed)

## 6) Known residual risks and compensating controls
Residual risk:
- A local admin could still alter files and recompute hashes.

Compensating controls recommended for compliance-grade clearance:
- Store JSONL log on append-only/WORM storage or remote immutable sink.
- Add digital signature with key held outside endpoint.
- Restrict local file permissions to least privilege.
- Add periodic off-host backup and monitoring alerts for log gaps.
- Document retention and review cadence (daily/weekly control).

## 7) Clearance position
Current design is suitable for strong internal audit and operational governance.
For strict regulatory clearance, add immutable storage + signing + access governance.
