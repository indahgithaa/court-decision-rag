# Keputusan pilot reranking

Tanggal eksperimen: 29 September 2026

## Setup

- Dataset development: `exploration_20`, 80 pertanyaan approved.
- Kandidat: dense retrieval top-50 dengan `intfloat/multilingual-e5-small`.
- Reranker: BM25 corpus-level dan weighted reciprocal-rank fusion.
- Parameter: `k1=1.2`, `b=0.75`, konstanta RRF `60`.
- Bobot yang diuji identik untuk fixed-size dan SAC.
- Metrik seleksi: rata-rata Hit@5 kedua strategi, lalu rata-rata MRR@5,
  lalu bobot dense yang lebih besar.

## Hasil

| Dense | BM25 | Fixed Hit@5 | SAC Hit@5 | SAC - fixed | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 0.4750 | 0.4750 | +0.0000 | [-0.1250, +0.1250] |
| 0.75 | 0.25 | 0.4250 | 0.4875 | +0.0625 | [-0.0625, +0.1875] |
| 0.50 | 0.50 | 0.4375 | 0.4125 | -0.0250 | [-0.1375, +0.0875] |
| 0.25 | 0.75 | 0.4375 | 0.4250 | -0.0125 | [-0.1250, +0.1125] |
| 0.00 | 1.00 | 0.3625 | 0.3875 | +0.0250 | [-0.0875, +0.1375] |

Candidate ceiling Hit@50 adalah `0.9250` untuk fixed-size dan `0.8750` untuk
SAC. Tidak ada selisih SAC-minus-fixed pada sweep ini yang interval kepercayaan
95%-nya tidak melintasi nol.

## Keputusan

Pertahankan dense-only (`dense_weight=1.0`) sebagai operating point. Fusion
BM25 tidak meningkatkan rata-rata Hit@5 kedua strategi. Bobot `0.75` memang
menaikkan SAC dari `0.4750` menjadi `0.4875`, tetapi menurunkan fixed-size dari
`0.4750` menjadi `0.4250`; karena itu konfigurasi tersebut tidak dipilih.

Keputusan ini merupakan tuning pada development set, bukan klaim hasil akhir.
Konfigurasi retrieval tidak boleh diubah setelah evaluasi holdout dimulai.
