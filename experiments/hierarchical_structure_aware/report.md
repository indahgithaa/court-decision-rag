# Hierarchical Structure-Aware Chunking untuk Legal RAG

## Status dan batas klaim

Eksperimen empat metode selesai pada korpus 200 putusan dan 800 pertanyaan. Seluruh hasil bersifat eksploratif: benchmark yang sama telah dipakai dalam eksperimen terdahulu, sehingga ini bukan holdout blind atau bukti konfirmatori independen.

Benchmark aktual hanya memiliki lima label pertanyaan: `amar_putusan`, `identitas_terdakwa`, `pertimbangan_hukum`, `riwayat_dakwaan`, dan `riwayat_penahanan`. Karena itu hanya ada tiga bagian lain di luar dua fokus, bukan empat.

## Landasan ilmiah

- Chen et al. (EMNLP 2024), *Dense X Retrieval*, DOI [10.18653/v1/2024.emnlp-main.845](https://doi.org/10.18653/v1/2024.emnlp-main.845), menunjukkan pentingnya granularitas unit dense retrieval dan evaluasi dengan budget konteks tetap.
- Lu et al. (ACL 2026), *HiChunk*, DOI [10.18653/v1/2026.acl-long.1372](https://doi.org/10.18653/v1/2026.acl-long.1372), memisahkan struktur hierarkis dari retrieval granular serta melakukan penggabungan ke parent dengan kendala budget. Komponen fine-tuned LLM dan Auto-Merge adaptifnya tidak dipakai karena berada di luar desain.
- Elchafei et al. (SemEval 2026), *H-RAG*, DOI [10.18653/v1/2026.semeval-1.155](https://doi.org/10.18653/v1/2026.semeval-1.155), mendukung child-first retrieval, max-child parent scoring, dan parent-context reconstruction. Komponen sparse, reranker, dan query rewritingnya tidak dipakai.
- Sarthi et al. (ICLR 2024), *RAPTOR*, [OpenReview](https://openreview.net/forum?id=GN921JHCRw), menjadi referensi konseptual multi-granular retrieval; summarization dan clustering rekursif tidak diimplementasikan.

## Rancangan dan konfigurasi beku

| Kode | Metode | Unit retrieval | Konteks yang dikembalikan |
|---|---|---|---|
| B1 | Fixed-size asli | 500 kata, overlap 100 | chunk yang sama |
| B2 | SAC asli | 150 kata, overlap 30, section-bound | chunk yang sama |
| B3 | Hierarchical | child 150/30 dalam parent linear 500/100 | parent unik |
| M1 | Hierarchical SAC | child 150/30 dalam parent section-bound 500/100 | parent unik |

Semua metode memakai `intfloat/multilingual-e5-small`, independent embedding, prefix E5, exact cosine search, top-50 candidates, tanpa BM25, reranker, contextual embedding, summary, atau label gold saat retrieval. Parent B3 dan M1 diberi skor maksimum child; konteks parent dideduplikasi dan dibatasi 2.500 kata per query.

## Audit baseline historis

Angka historis yang diwajibkan tetap dicatat tanpa perubahan:

| Baseline historis | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| Fixed-Size + Independent | 0.6350 | 0.3638 | 0.3489 |
| Structure-Aware + Independent | 0.6425 | 0.3935 | 0.4414 |

Audit repository menemukan inkonsistensi provenance: angka Fixed 0.6350/0.3638/0.3489 tersimpan pada eksperimen 150/30, sedangkan dokumen seleksi baseline membekukan Fixed 500/100. Eksperimen baru mengikuti konfigurasi baseline beku 500/100 untuk B1 dan melaporkan hasilnya terpisah; angka historis di atas tidak ditimpa atau dicampur.

## Audit implementasi

| Metode | Child/chunk | Parent | Maks token encoder | >512 | Semua kata tercakup |
|---|---:|---:|---:|---:|---|
| B1 | 3527 | 0 | 949 | 3380 | ya |
| B2 | 12206 | 0 | 327 | 0 | ya |
| B3 | 13828 | 3527 | 317 | 0 | ya |
| M1 | 13811 | 4595 | 320 | 0 | ya |

B3 dan M1 divalidasi memiliki mapping parent-child lengkap. Parent M1 tidak pernah melintasi section. Over-limit B1 adalah konsekuensi baseline 500 kata yang dipertahankan; B2/B3/M1 diperiksa terhadap batas aktual encoder.

## Hasil keseluruhan

| Metode | R@1 | R@3 | R@5 | R@10 | MRR@1 | MRR@3 | MRR@5 | MRR@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.2437 | 0.4612 | 0.5437 | 0.6175 | 0.2437 | 0.3383 | 0.3574 | 0.3671 | 0.2437 | 0.3223 | 0.3468 | 0.3715 |
| B2 | 0.2737 | 0.4800 | 0.6425 | 0.7362 | 0.2737 | 0.3560 | 0.3935 | 0.4070 | 0.2737 | 0.3806 | 0.4414 | 0.4684 |
| B3 | 0.2100 | 0.4625 | 0.6225 | 0.7612 | 0.2100 | 0.3183 | 0.3543 | 0.3736 | 0.2100 | 0.2874 | 0.3360 | 0.3797 |
| M1 | 0.2700 | 0.4713 | 0.6400 | 0.7400 | 0.2700 | 0.3506 | 0.3895 | 0.4034 | 0.2700 | 0.3756 | 0.4377 | 0.4662 |

## Perubahan absolut dan CI 95% pada @5

CI memakai paired cluster bootstrap 10.000 kali pada unit dokumen. nDCG lintas chunker tetap deskriptif karena jumlah chunk relevan yang overlap dapat berbeda antarstrategi.

| Kontras | ΔRecall@5 [CI] | ΔMRR@5 [CI] | ΔnDCG@5 [CI] |
|---|---:|---:|---:|
| B2 − B1 | +0.0988 [+0.0550, +0.1412] | +0.0361 [+0.0053, +0.0663] | +0.0947 [+0.0655, +0.1239] |
| B3 − B1 | +0.0788 [+0.0338, +0.1237] | -0.0031 [-0.0366, +0.0293] | -0.0108 [-0.0394, +0.0186] |
| M1 − B1 | +0.0963 [+0.0512, +0.1388] | +0.0321 [+0.0012, +0.0623] | +0.0909 [+0.0619, +0.1202] |
| B3 − B2 | -0.0200 [-0.0550, +0.0163] | -0.0393 [-0.0650, -0.0138] | -0.1055 [-0.1271, -0.0838] |
| M1 − B2 | -0.0025 [-0.0163, +0.0112] | -0.0040 [-0.0100, +0.0020] | -0.0038 [-0.0092, +0.0016] |
| M1 − B3 | +0.0175 [-0.0187, +0.0537] | +0.0352 [+0.0091, +0.0612] | +0.1017 [+0.0796, +0.1234] |

## Metrik @5 per section

| Section | N | Metode | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---|---:|---:|---:|
| `amar_putusan` | 200 | B1 | 0.3100 | 0.1742 | 0.1962 |
| `amar_putusan` | 200 | B2 | 0.7900 | 0.3388 | 0.4502 |
| `amar_putusan` | 200 | B3 | 0.6650 | 0.3621 | 0.4127 |
| `amar_putusan` | 200 | M1 | 0.7700 | 0.3258 | 0.4353 |
| `identitas_terdakwa` | 194 | B1 | 0.7320 | 0.5707 | 0.6109 |
| `identitas_terdakwa` | 194 | B2 | 0.9485 | 0.9357 | 0.9389 |
| `identitas_terdakwa` | 194 | B3 | 0.5979 | 0.3698 | 0.4263 |
| `identitas_terdakwa` | 194 | M1 | 0.9433 | 0.9347 | 0.9369 |
| `pertimbangan_hukum` | 200 | B1 | 0.5900 | 0.3152 | 0.1854 |
| `pertimbangan_hukum` | 200 | B2 | 0.1750 | 0.0671 | 0.0399 |
| `pertimbangan_hukum` | 200 | B3 | 0.7150 | 0.4705 | 0.2345 |
| `pertimbangan_hukum` | 200 | M1 | 0.1750 | 0.0652 | 0.0381 |
| `riwayat_dakwaan` | 200 | B1 | 0.5500 | 0.3815 | 0.4053 |
| `riwayat_dakwaan` | 200 | B2 | 0.6850 | 0.2606 | 0.3649 |
| `riwayat_dakwaan` | 200 | B3 | 0.5200 | 0.2208 | 0.2768 |
| `riwayat_dakwaan` | 200 | M1 | 0.7000 | 0.2603 | 0.3684 |
| `riwayat_penahanan` | 6 | B1 | 0.5000 | 0.1722 | 0.2530 |
| `riwayat_penahanan` | 6 | B2 | 0.0000 | 0.0000 | 0.0000 |
| `riwayat_penahanan` | 6 | B3 | 0.3333 | 0.1667 | 0.2103 |
| `riwayat_penahanan` | 6 | M1 | 0.0000 | 0.0000 | 0.0000 |

## Konteks yang dikembalikan dengan budget setara

Evidence hit di bawah mengukur apakah konteks akhir (chunk untuk B1/B2, parent untuk B3/M1) memuat evidence gold dalam budget 2.500 kata. Ini dipisahkan dari metrik child retrieval di atas.

| Metode | Evidence hit | Rata-rata kata | Median kata | Rata-rata unit |
|---|---:|---:|---:|---:|
| B1 | 0.5463 | 2459.7 | 2500.0 | 5.42 |
| B2 | 0.7688 | 2469.8 | 2478.0 | 20.37 |
| B3 | 0.8738 | 2461.3 | 2500.0 | 5.03 |
| M1 | 0.8425 | 2456.0 | 2470.0 | 9.54 |

## Analisis Pertimbangan Hukum

Recall@5: B1=0.5900, B2=0.1750, B3=0.7150, dan M1=0.1750. Nilai tertinggi adalah B3 (0.7150). Delta M1 terhadap B1 -0.4150, terhadap B2 +0.0000, dan terhadap B3 -0.5400. Di bawah budget konteks yang sama, evidence hit B1/B2/B3/M1 masing-masing 0.5900/0.3450/0.9200/0.7200. Perbedaan ini menunjukkan bahwa child ranking dan keberhasilan menyediakan parent context adalah dua outcome yang berbeda.

## Analisis Riwayat Penahanan

Recall@5: B1=0.5000, B2=0.0000, B3=0.3333, dan M1=0.0000. Nilai tertinggi adalah B1 (0.5000). Delta M1 terhadap B1 -0.5000, terhadap B2 +0.0000, dan terhadap B3 -0.3333. Di bawah budget konteks yang sama, evidence hit B1/B2/B3/M1 masing-masing 0.5000/0.0000/1.0000/0.0000. Perbedaan ini menunjukkan bahwa child ranking dan keberhasilan menyediakan parent context adalah dua outcome yang berbeda.

Strata Riwayat Penahanan hanya berisi enam pertanyaan. Satu query mengubah Recall sebesar 0,1667, sehingga pola ini terlalu rapuh untuk klaim umum atau klaim signifikansi section-spesifik.

## Error analysis @5

| Metode | Sukses | Salah dokumen | Salah section | Evidence rank 6–50 | Tidak ada di top-50 |
|---|---:|---:|---:|---:|---:|
| B1 | 435 | 66 | 231 | 30 | 38 |
| B2 | 514 | 2 | 140 | 48 | 96 |
| B3 | 498 | 1 | 151 | 71 | 79 |
| M1 | 512 | 1 | 150 | 45 | 92 |

Kategori error di atas eksklusif dengan prioritas salah dokumen, lalu salah section, lalu posisi evidence. Karena itu jumlah `Tidak ada di top-50` bukan total seluruh query tanpa evidence di top-50; total tersebut dilaporkan pada tabel ranking berikut.

## Ranking evidence relevan

| Metode | Evidence ditemukan top-50 | Tidak ditemukan top-50 | Median rank jika ditemukan | Mean rank | P90 rank |
|---|---:|---:|---:|---:|---:|
| B1 | 619 | 181 | 3.0 | 7.20 | 23 |
| B2 | 633 | 167 | 3.0 | 4.06 | 8 |
| B3 | 690 | 110 | 3.0 | 5.39 | 11 |
| M1 | 637 | 163 | 3.0 | 4.29 | 9 |

## Testing dan validasi

Validasi runtime lulus untuk exact source offsets, cakupan seluruh kata sumber, keunikan ID, qrels positif untuk semua 800 query, mapping parent-child, deduplikasi parent, budget konteks, dan batas section M1. Full repository test suite lulus: 122 passed, 0 failed, 0 errors.

## Kesimpulan ilmiah

Secara agregat Recall@5 M1 menurun dari 0.6425 (B2) menjadi 0.6400; delta -0.0025 dengan CI 95% [-0.0163, +0.0112] (CI melintasi nol). Pada Pertimbangan Hukum, Recall@5 M1 tidak berubah dari 0.1750 menjadi 0.1750. Jadi M1 tidak memperbaiki objective child retrieval yang menjadi kriteria utama dan belum didukung sebagai pengganti SAC murni. Namun evidence hit konteks ber-budget sama naik dari 0.7688 menjadi 0.8425 secara keseluruhan dan dari 0.3450 menjadi 0.7200 pada Pertimbangan Hukum. Ini adalah manfaat context delivery, bukan peningkatan child ranking, dan belum diberi CI khusus. B3 menunjukkan bahwa hierarchy linear dapat membantu Pertimbangan, tetapi tabel per section menunjukkan trade-off besar pada bagian terstruktur. Hasil tetap eksploratif dan memerlukan pertanyaan baru atau korpus eksternal.

## Keterbatasan

- Dataset QA dibuat deterministik dari XML dan belum merupakan adjudikasi manusia independen.
- Benchmark 200 dokumen telah dilihat pada eksperimen sebelumnya; CI mengukur ketidakpastian sampling internal, bukan mengubahnya menjadi holdout blind.
- Hanya enam query Riwayat Penahanan.
- Parent 500 kata adalah konteks sumber asli, bukan ringkasan; hasil tidak menguji RAPTOR atau HiChunk lengkap.
- MRR dan nDCG menggunakan qrels chunk-spesifik; Recall evidence dan context evidence hit lebih langsung sebanding lintas chunker.
- Baseline B1 500 kata dapat melampaui 512 token subword dan ditrunkasi encoder sesuai perilaku baseline historis.

## Reproduksi

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv\Scripts\python.exe -m pytest experiments\hierarchical_structure_aware\test_hierarchical_chunking.py -q
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m experiments.hierarchical_structure_aware.run_experiment
```

Untuk memvalidasi ulang artefak lokal yang sudah lengkap tanpa menimpa eksperimen lain, tambahkan `--resume`.
