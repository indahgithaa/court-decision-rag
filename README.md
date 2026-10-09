# Court Decision RAG

Repository penelitian retrieval dokumen putusan pengadilan Indonesia.

## Eksperimen aktif: Hierarchical SAC

Arah penelitian aktif membandingkan tepat tiga konfigurasi dengan batas chunk
canonical yang sama:

- B1 - Recursive Character Chunking.
- B2 - Summary-Augmented Chunking dengan satu ringkasan dokumen.
- M1 - ringkasan dokumen dan ringkasan bagian.

Konfigurasi beku berada di `configs/hierarchical_sac_2026.yaml`. Jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\27_run_hierarchical_sac.py --config configs\hierarchical_sac_2026.yaml --stage all
```

Dokumentasi metode, audit dataset, laporan Indonesia, tabel, dan grafik ditulis ke
`experiments/hierarchical_sac_2026/` dan
`experiments/results/hierarchical_sac_2026/`.

Corpus aktif adalah 200 dokumen XML Indo-Law yang sudah dinormalisasi. Tidak ada PDF
di repository saat ini. Benchmark lama juga belum tervalidasi manusia dan holdout
historis telah dibuka, sehingga hasil baru berstatus development eksploratori sampai
tersedia holdout baru dengan anotasi manusia.

## Pengujian

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Artefak historis tetap dipertahankan dan tidak dipakai sebagai hasil eksperimen baru.
