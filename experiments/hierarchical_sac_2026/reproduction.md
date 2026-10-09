# Reproduksi eksperimen

## Lingkungan

- Python 3.12
- Dependensi: `pip install -r requirements-research.txt`
- Model lokal: `Irvan14/t5-small-indonesian-summarization` revisi
  `65f2d2be6dac02a57444535aca909c5d4606de8a` dan `intfloat/multilingual-e5-small`
  revisi `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- API berbayar tidak digunakan.
- Jika PyTorch diblokir Windows Smart App Control, runner memakai ekspor ONNX FP32
  resmi revisi `8d923955b027282ba975c0a4c825486c9ca4c490` dengan checksum
  `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665` dan mewajibkan uji equivalence terhadap
  cache PyTorch sebelum mencampur backend.

## Perintah

```powershell
.\.venv\Scripts\python.exe scripts\18_restore_indolaw_from_manifest.py
.\.venv\Scripts\python.exe scripts\12_prepare_indolaw_corpus.py
.\.venv\Scripts\python.exe scripts\27_run_hierarchical_sac.py --config configs\hierarchical_sac_2026.yaml --stage all
.\.venv\Scripts\python.exe -m pytest
```

Setiap tahap memakai hash sumber dan cache. Gunakan `--force` hanya jika memang ingin
menghasilkan ulang ringkasan/embedding dengan konfigurasi yang sama. Seed statistik
adalah 42.
