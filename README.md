# Swarm Forge

Swarm Forge adalah experimental engineering project dan fondasi untuk mengeksplorasi orkestrasi multi-agent. Proyek ini mempelajari bagaimana reliability, penegakan kebijakan yang deterministik, auditability, dan task lifecycle dapat dirancang serta diuji sebelum membangun sistem orkestrasi yang lebih lengkap.

## Current capabilities

- Kontrak task, policy, dan event berbasis Pydantic.
- Model task lifecycle, validasi transisi state, penyimpanan SQLite, serta workflow validation.
- Policy gate deterministik dengan aturan fail-closed.
- Audit log SQLite berantai SHA-256, termasuk pemeriksaan integritas chain untuk mendeteksi perubahan data.
- Kontrak event lifecycle dengan validasi bentuk dan transisi.
- Primitive persistensi transactional lifecycle outbox: fungsi menerima koneksi dan transaksi milik pemanggil, dengan dukungan idempotency serta validasi manifest event.

Kemampuan tersebut menggambarkan implementasi pada checkpoint repositori, bukan jaminan kesiapan produksi.

## Technology stack

Stack yang terverifikasi dari `pyproject.toml` dan kode: Python 3.12+, Pydantic 2, dan SQLite melalui modul standar `sqlite3`. Tool pengembangan yang dideklarasikan: uv build backend, pytest, Ruff, dan mypy.

## Development

Dengan [uv](https://docs.astral.sh/uv/) terpasang:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
```

## Status and limitations

Checkpoint proyek dibekukan pada branch `feat/task-workflow-state-machine`, commit `58d310103f2eb75a2b795c361a40ed4a63b16ccc` (`feat: add transactional lifecycle outbox persistence`). Integrasi lifecycle yang secara atomik menghubungkan perubahan state dengan pencatatan outbox belum selesai, dan outbox dispatcher belum diimplementasikan. Proyek ini belum production-ready dan belum mengklaim sebagai platform eksekusi multi-agent penuh. Pengembangan berikutnya bergantung pada kebutuhan nyata, bukan penambahan infrastruktur tanpa justifikasi.

## Development disclosure

Developed with Codex assistance.
