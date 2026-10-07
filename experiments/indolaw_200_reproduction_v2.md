# Reproduksi retrieval Indo-Law 200 — schema v2

Tanggal eksekusi: 7 Oktober 2026  
Commit dasar: `62f55d60aac0d82308c10876065e3de3b13d1f40` (`git_dirty=true`)  
Schema metrik: `retrieval-v2-evidence-recall`

## Ruang lingkup dan status inferensi

Eksperimen ini menggunakan 200 putusan Indo-Law dalam XML ternormalisasi:
40 dokumen development dan 160 dokumen holdout, dikelompokkan berdasarkan
pengadilan tanpa overlap. Batas section berasal dari corpus. Karena itu, hasil
ini adalah **oracle-structure robustness experiment**, bukan evaluasi ekstraksi
PDF atau deteksi struktur otomatis.

Split holdout pernah dibuka pada eksperimen lama sebelum schema v2 dan aturan
pemilihan desain dibekukan. Hasil holdout di bawah adalah **corrected
exploratory analysis**, bukan konfirmasi blind. Klaim konfirmatori memerlukan
holdout baru atau validasi eksternal.

## Data dan environment

- Sumber dipin ke commit Indo-Law
  `67340338adbc63021630947d905627eef0d7e43f`.
- Semua 200 XML direstorasi sesuai path manifest dan diverifikasi byte serta
  SHA-256.
- Pertanyaan dibuat deterministik dari XML: 160 development dan 640 holdout.
  Status `approved` dihasilkan otomatis; dataset belum melalui adjudikasi manusia.
- Model embedding: `intfloat/multilingual-e5-small` dari cache commit terpin.
- Python 3.12.10; NumPy 2.5.3; sentence-transformers 5.7.0; PyTorch 2.14.1;
  Transformers 5.19.0; device CPU; seed 42; retrieval corpus-wide; top-50.
- Recall mengukur coverage `evidence_id` yang sama pada kedua strategi dan
  mendeduplikasi overlapping chunk untuk bukti yang sama. MRR dan nDCG tetap
  berbasis ranking chunk.

## Seleksi development

Aturan yang dibekukan: maksimalkan nDCG@5, lalu MRR@5, Recall@5, lalu minimalkan
jumlah chunk.

| Konfigurasi | Chunk | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_w150_o30` | 2341 | 0.6562 | 0.3622 | 0.3492 | 0.8063 | 0.8938 |
| `fixed_w300_o60` | 1173 | 0.6562 | 0.3914 | 0.3779 | 0.7562 | 0.8688 |
| `fixed_w500_o100` | 710 | 0.6500 | 0.4315 | 0.4168 | 0.7438 | 0.9187 |
| `sac_w150_o30_s0` | 2457 | 0.6687 | 0.4146 | 0.4587 | 0.7250 | 0.8125 |
| `sac_w300_o60_s0` | 1346 | 0.6500 | 0.4322 | 0.4554 | 0.7750 | 0.8938 |
| `sac_w500_o100_s0` | 923 | 0.6125 | 0.4255 | 0.4441 | 0.6875 | 0.8750 |

Desain terpilih: `fixed_w500_o100` dan `sac_w150_o30_s0`.

## Holdout lama yang dikoreksi (eksploratif)

Evaluasi mencakup 640 pertanyaan dalam 160 dokumen. Interval kepercayaan 95%
dihitung dengan 10.000 paired cluster bootstrap pada unit dokumen.

| Metrik | Fixed | SAC | Selisih SAC - fixed | 95% CI |
|---|---:|---:|---:|---:|
| Recall@5 | 0.5406 | 0.6375 | +0.0969 | [+0.0484, +0.1437] |
| MRR@5 | 0.3574 | 0.3890 | +0.0316 | [-0.0032, +0.0647] |
| nDCG@5 | 0.3476 | 0.4380 | +0.0904 | [+0.0570, +0.1223] |
| Recall@10 | 0.6234 | 0.7406 | +0.1172 | [+0.0703, +0.1625] |
| MRR@10 | 0.3686 | 0.4038 | +0.0352 | [+0.0015, +0.0673] |
| nDCG@10 | 0.3743 | 0.4678 | +0.0935 | [+0.0623, +0.1231] |

Efek @5 sangat heterogen menurut bagian. Recall SAC lebih tinggi pada
`amar_putusan` (0.7812 vs 0.3000) dan `identitas_terdakwa` (0.9613 vs 0.7355),
tetapi lebih rendah pada `pertimbangan_hukum` (0.1625 vs 0.5750). Karena itu,
hasil tidak mendukung klaim bahwa SAC unggul merata pada semua bagian dokumen.
Strata `riwayat_penahanan` hanya memiliki lima pertanyaan dan tidak layak
ditafsirkan sendiri.

Latency yang tersimpan adalah embedding query batch yang diamortisasi ditambah
search. Nilainya hanya diagnostik, bukan latency online atau end-to-end.

## Artefak dan hash

Artefak besar berada di direktori yang diabaikan Git dan harus dipertahankan
melalui registry/bundle terpisah sebelum hasil dipakai dalam skripsi.

| Artefak | SHA-256 |
|---|---|
| `experiments/results/indolaw_200_development_design_grid.json` | `1229df00177fa6316a8b532883f4e5c96fba0bca3c41f9e8f6abeb330f19ea60` |
| `experiments/results/indolaw_200_development_grid_run_manifest.json` | `ee952bb41c09f0ee83cb99be77d3523d39bbde968d550d4d706618d73fed3427` |
| `experiments/results/indolaw_200_holdout_evaluation_v2.json` | `23df55192655050cc16242c1eff1cc3efdf82d3a9d11546e4e1fc3fb76576935` |
| `experiments/results/indolaw_200_holdout_grid_run_manifest.json` | `8addb35f6738c127a80a915cb1e7b9c575249786c43137ea5020f80408eeb9f4` |

Manifest run juga menyimpan hash questions, chunks, rankings, model/package,
platform, seed, commit dasar, dan hash source evaluator. Hasil v2 tidak memuat
Hit@K.

## Keputusan ilmiah

1. Pakai hasil ini untuk debugging dan pembahasan robustness oracle-structure,
   dengan label eksploratif yang eksplisit.
2. Jangan pakai hasil ini sebagai bukti efektivitas detector struktur pada PDF.
3. Jangan menyebut holdout ini blind/final.
4. Sebelum klaim utama, validasi pertanyaan/evidence oleh manusia, bekukan
   environment, dan jalankan PDF holdout atau validasi eksternal yang belum
   pernah dibuka.
