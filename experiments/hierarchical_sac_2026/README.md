# Hierarchical Summary-Augmented Chunking 2026

Workspace ini hanya untuk eksperimen baru B1/B2/M1. Artefak historis di
direktori lain tidak diubah dan nilainya tidak dipakai sebagai hasil baru.

Konfigurasi beku: `configs/hierarchical_sac_2026.yaml`.

Lingkungan eksperimen dibekukan di `requirements-research.txt`.

Jalankan seluruh pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\27_run_hierarchical_sac.py --config configs\hierarchical_sac_2026.yaml --stage all
```

Tahap dapat dilanjutkan secara terpisah dengan `--stage audit`, `summaries`,
`chunks`, `representations`, `embeddings`, `evaluate`, atau `report`.
Pipeline menolak cache yang tidak sesuai konfigurasi atau hash sumber.

Status ilmiah corpus aktif adalah normalized-text development experiment.
Corpus berisi XML Indo-Law, bukan PDF, dan benchmark belum divalidasi manusia.

Hasil eksekusi terakhir dan status test dicatat di `verification.md`.
