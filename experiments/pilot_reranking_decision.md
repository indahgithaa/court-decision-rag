# Keputusan pilot reranking

Tanggal eksperimen: 29 September 2026

## Setup

- Dataset development: `exploration_20`, 80 pertanyaan approved.
- Kandidat: dense retrieval top-50 dengan `intfloat/multilingual-e5-small`.
- Reranker: BM25 corpus-level dan weighted reciprocal-rank fusion.
- Parameter: `k1=1.2`, `b=0.75`, konstanta RRF `60`.
- Bobot yang diuji identik untuk fixed-size dan SAC.
- Urutan seleksi: rata-rata NDCG@5, MRR@5, Recall@5, lalu bobot dense yang
  lebih besar.

## Hasil

| Dense | BM25 | Fixed Recall@5 | SAC Recall@5 | Fixed MRR@5 | SAC MRR@5 | Fixed NDCG@5 | SAC NDCG@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,00 | 0,00 | 0,4500 | 0,4750 | 0,2848 | 0,2821 | 0,3102 | 0,3294 |
| 0,75 | 0,25 | 0,4062 | 0,4875 | 0,2690 | 0,2873 | 0,2884 | 0,3365 |
| 0,50 | 0,50 | 0,4125 | 0,4125 | 0,2617 | 0,2419 | 0,2875 | 0,2845 |
| 0,25 | 0,75 | 0,4125 | 0,4250 | 0,2540 | 0,2048 | 0,2863 | 0,2593 |
| 0,00 | 1,00 | 0,3375 | 0,3875 | 0,2267 | 0,1785 | 0,2479 | 0,2299 |

Candidate Recall@50 adalah 0,9000 untuk fixed-size dan 0,8688 untuk SAC pada
operating point dense-only.

## Keputusan

Pertahankan dense-only (`dense_weight=1.0`). Konfigurasi ini memiliki rata-rata
NDCG@5 tertinggi untuk kedua strategi. Fusion dengan bobot dense 0,75 menaikkan
NDCG@5 SAC, tetapi menurunkan NDCG@5 fixed-size dan rata-rata gabungannya.

Keputusan ini merupakan tuning pada development set, bukan klaim hasil akhir.
