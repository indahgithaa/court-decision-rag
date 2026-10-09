# Verifikasi eksekusi

Verifikasi terakhir dilakukan pada 9 Oktober 2026 (Asia/Jakarta).

## Pipeline

Perintah berikut selesai dengan exit code 0:

```powershell
.\.venv\Scripts\python.exe scripts\27_run_hierarchical_sac.py --config configs\hierarchical_sac_2026.yaml --stage all
```

Tahap audit, cache ringkasan, chunk canonical, representasi, embedding, evaluasi,
dan pelaporan semuanya berhasil. Evaluasi ulang menghasilkan Recall@5 yang sama:
B1 0,4979; B2 0,5048; dan M1 0,4828.

## Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=experiments\results\hierarchical_sac_2026\pytest.xml
```

- Dikoleksi: 126.
- Lulus: 125.
- Gagal/error: 0.
- Dilewati: 1 modul test contextual embedding historis.

Satu modul dilewati karena Windows Application Control memblokir
`torch_python.dll` dengan WinError 4551. Kondisi itu ditangani hanya untuk error
kebijakan tersebut; error impor lain tetap dinaikkan. Sepuluh test khusus pipeline
Hierarchical SAC semuanya lulus. Backend ONNX eksperimen juga melewati pemeriksaan
kesetaraan numerik terhadap delapan embedding referensi PyTorch.

## Visualisasi

Enam SVG hasil eksperimen berhasil dirender menjadi PNG. Grafik interval
Recall@5 dan Recall@5 per bagian diperiksa secara visual; label, legenda, nilai,
dan interval tampil lengkap tanpa elemen terpotong.
