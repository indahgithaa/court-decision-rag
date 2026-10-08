# Hasil eksperimen faktorial chunking × contextualized embeddings

## Ringkasan hasil

Pada konfigurasi yang diuji, **contextualized chunk embeddings tidak
meningkatkan retrieval**. Dibanding embedding independen, contextualization
menurunkan Recall@5, MRR@5, dan nDCG@5 baik pada fixed-size maupun SAC; seluruh
CI 95% untuk ketiga penurunan tersebut tidak melintasi nol. Kombinasi yang
terbaik secara agregat adalah **SAC + independent** untuk MRR@5 dan nDCG@5,
sedangkan Recall@5-nya tidak berbeda meyakinkan dari Fixed + independent.

Hasil ini berlaku untuk implementasi dan encoder yang diuji, bukan bukti bahwa
semua bentuk contextualized embedding atau late chunking selalu lebih buruk.

## Setup

- Desain: faktorial 2×2, yaitu `fixed_size`/`structure_aware` ×
  `independent`/`contextual`.
- Corpus: 200 putusan Indo-Law dan 800 pertanyaan dari 200 dokumen.
- Chunking: maksimum 150 kata dan overlap 30 kata pada Fixed dan SAC. SAC
  mempertahankan batas bagian XML; Fixed mengabaikan batas bagian.
- Encoder seluruh lengan: `intfloat/multilingual-e5-small`, prefix E5 yang sama,
  embedding ternormalisasi, dan exact cosine search pada seluruh corpus.
- Independent: setiap chunk di-encode sendiri. Contextual: token dokumen
  di-encode dalam window 512 token dengan overlap 128 token, kemudian token
  pada span chunk di-mean-pool (*windowed late chunking*). Chunk, teks, dan
  offset dalam setiap pasangan Independent–Contextual identik.
- Retrieval: top-50; hasil utama di bawah dilaporkan pada cutoff 5.
- Recall memakai unit evidence bersama. nDCG memakai chunk-level qrels yang
  spesifik untuk Fixed dan SAC.
- Setiap query hanya memiliki satu evidence unit, sehingga Recall@K pada
  eksperimen ini secara numerik sama dengan evidence Hit@K. Seluruh qrel
  positif juga memakai grade 2; Fixed memiliki 1.427 chunk positif dan SAC
  1.345. Karena itu nDCG lintas chunker dan interaksinya diperlakukan sebagai
  diagnosis deskriptif, bukan bukti faktorial primer.
- Ketidakpastian: 10.000 paired percentile cluster bootstrap, unit resampling
  dokumen, seed 42.
- Status inferensi: evaluasi komparatif eksploratif pada satu corpus, bukan
  pengujian konfirmatori pada blind holdout.

## Hasil agregat @5

| Kondisi | Recall@5 | MRR@5 | nDCG@5 | DRM@5 ↓ | Document hit@5 |
|---|---:|---:|---:|---:|---:|
| Fixed + independent | 0.6350 | 0.3638 | 0.3489 | 0.0620 | 0.9962 |
| Fixed + contextual | 0.3887 | 0.2300 | 0.2283 | 0.2442 | 0.9425 |
| SAC + independent | **0.6425** | **0.3935** | **0.4414** | 0.1028 | **0.9975** |
| SAC + contextual | 0.4000 | 0.2652 | 0.2667 | 0.2750 | 0.9363 |

DRM adalah proporsi hasil top-k yang berasal dari dokumen selain dokumen gold;
lebih rendah lebih baik. Contextualization tidak hanya menurunkan tiga metrik
utama, tetapi juga meningkatkan DRM secara besar pada kedua chunker.

## Kontras terencana @5

Delta selalu mengikuti `kondisi pertama − kondisi pembanding`. Interaksi
didefinisikan sebagai `(SAC contextual − SAC independent) − (Fixed
contextual − Fixed independent)`.

| Kontras | Δ Recall@5 (CI 95%) | Δ MRR@5 (CI 95%) | Δ nDCG@5 (CI 95%) |
|---|---:|---:|---:|
| Contextual − Independent, dalam Fixed | -0.2462 [-0.2863, -0.2062] | -0.1338 [-0.1652, -0.1030] | -0.1206 [-0.1469, -0.0951] |
| Contextual − Independent, dalam SAC | -0.2425 [-0.2737, -0.2100] | -0.1284 [-0.1513, -0.1056] | -0.1748 [-0.1962, -0.1536] |
| SAC − Fixed, pada Independent | +0.0075 [-0.0275, +0.0425] | +0.0297 [+0.0024, +0.0567] | +0.0925 [+0.0698, +0.1150] |
| SAC − Fixed, pada Contextual | +0.0112 [-0.0138, +0.0375] | +0.0351 [+0.0175, +0.0527] | +0.0383 [+0.0218, +0.0546] |
| SAC contextual − Fixed independent | -0.2350 [-0.2750, -0.1950] | -0.0986 [-0.1280, -0.0701] | -0.0823 [-0.1086, -0.0572] |
| Efek marginal Contextual | -0.2444 [-0.2744, -0.2137] | -0.1311 [-0.1534, -0.1092] | -0.1477 [-0.1680, -0.1279] |
| Efek marginal SAC | +0.0094 [-0.0138, +0.0331] | +0.0324 [+0.0161, +0.0489] | +0.0654 [+0.0501, +0.0803] |
| Interaksi chunking × embedding | +0.0037 [-0.0350, +0.0425] | +0.0054 [-0.0265, +0.0369] | -0.0541 [-0.0800, -0.0295] |

Efek interaksi Recall dan MRR tidak terpisah dari nol. Interaksi nDCG negatif:
contextualization mengurangi keunggulan ranking SAC lebih besar daripada pada
Fixed. Karena CI bootstrap adalah interval estimasi, istilah “berbeda” di sini
tidak dimaksudkan sebagai hasil uji hipotesis multipel formal.

## Recall@5 per bagian

| Bagian | N | Fixed + independent | Fixed + contextual | SAC + independent | SAC + contextual |
|---|---:|---:|---:|---:|---:|
| `amar_putusan` | 200 | 0.6400 | 0.1550 | **0.7900** | 0.1600 |
| `identitas_terdakwa` | 194 | 0.6495 | 0.6134 | **0.9485** | 0.8196 |
| `pertimbangan_hukum` | 200 | **0.7400** | 0.4500 | 0.1750 | 0.3900 |
| `riwayat_dakwaan` | 200 | 0.5150 | 0.3500 | **0.6850** | 0.2550 |
| `riwayat_penahanan` | 6 | **0.5000** | 0.1667 | 0.0000 | 0.0000 |

Heterogenitas bagian penting untuk interpretasi. SAC + independent unggul
deskriptif pada amar, identitas, dan riwayat dakwaan, tetapi jauh di bawah
Fixed + independent pada pertimbangan hukum. Contextualization menaikkan Recall
pertimbangan hukum SAC dari 0.1750 menjadi 0.3900, tetapi tetap di bawah kedua
lengan Fixed dan disertai penurunan besar pada bagian lain. Baris riwayat
penahanan hanya memiliki enam pertanyaan sehingga tidak layak ditafsirkan
sendiri.

## Interpretasi

1. **Contextualization yang diuji merugikan retrieval secara agregat.** Pola
   negatif hampir sama pada Fixed dan SAC untuk Recall dan MRR, sehingga masalah
   utamanya bukan batas chunk tertentu. Kenaikan DRM dan penurunan document hit
   menunjukkan bahwa representasi contextual menjadi kurang mampu membedakan
   provenance dokumen pada rank awal.
2. **SAC masih bermanfaat sebagai chunker pada embedding independen, tetapi
   manfaatnya bukan Recall agregat.** SAC meningkatkan MRR@5 dan nDCG@5 dengan
   CI positif, sementara selisih Recall@5 kecil dan CI-nya melintasi nol.
   Manfaat tersebut juga tidak seragam antarbagian.
3. **Kombinasi SAC + contextual bukan metode terbaik.** Dibanding baseline
   konvensional Fixed + independent, kombinasi itu turun 0.2350 pada Recall@5,
   0.0986 pada MRR@5, dan 0.0823 pada nDCG@5.
4. **Klaim yang didukung adalah konfigurasi-spesifik.** Eksperimen ini menguji
   encoder E5-small dengan konteks efektif 512 token. Encoder long-context,
   objective embedding yang memang dilatih untuk late chunking, strategi
   pooling lain, atau window assignment lain dapat memberi hasil berbeda dan
   memerlukan eksperimen baru yang dipisahkan dari hasil ini.

## Keterbatasan

- Corpus yang sama telah dipakai selama pengembangan metode, sehingga hasil
  bersifat eksploratif dan tidak mengukur generalisasi ke data baru.
- Pertanyaan dibuat secara deterministik dari XML dan berstatus approved secara
  otomatis; review manusia gold-standard belum selesai.
- SAC memakai section XML yang tersedia, sehingga hasil ini merupakan kondisi
  *oracle structure* dan bukan validasi deteksi struktur otomatis dari PDF.
- Hanya satu encoder, satu ukuran chunk, dan satu konfigurasi contextual window
  yang diuji. Desain ini tidak menguji late chunking dengan model long-context.
- Recall memakai evidence unit bersama, tetapi nDCG memakai qrels chunk yang
  spesifik strategi. Oleh sebab itu, nDCG harus dibaca bersama Recall dan MRR.
- Hasil per bagian tidak diberi interval perbandingan maupun koreksi
  multiplicity; seluruh pola per bagian bersifat deskriptif.
- Latency berasal dari sesi dan metode pengukuran batch yang berbeda waktu;
  nilainya bersifat diagnostik, bukan dasar klaim kecepatan antarlengan.

## Status evaluasi kualitas jawaban

**Faithfulness, Answer Relevance, dan BERTScore belum dijalankan.** Pada saat
eksperimen retrieval ini selesai, runner generasi jawaban masih berupa stub,
evaluator kualitas jawaban belum diimplementasikan, dependency Ragas/BERTScore
belum terpasang, dan provider/model/API key generatif maupun model judge belum
dibekukan. Karena itu laporan ini hanya menjawab kinerja retrieval; belum ada
dasar untuk menyatakan dampaknya terhadap kualitas jawaban RAG.

Full QA faktorial kelak membutuhkan 3.200 jawaban (800 pertanyaan × 4 lengan)
dengan generator, prompt, decoding, context budget, urutan eksekusi, dan judge
yang identik, lalu paired document-cluster bootstrap untuk kontras yang sama.

## Artefak

- Laporan lengkap:
  `experiments/results/indolaw_200_factorial_contextual_evaluation_verified.md`
- Hasil lengkap per-query dan bootstrap:
  `experiments/results/indolaw_200_factorial_contextual_evaluation_verified.json`
  (SHA-256
  `9576017abb870b4225931459db649420a957fd6539a56a26359fb93b076272ad`).
- Laporan Markdown terverifikasi memiliki SHA-256
  `29ffccd71ae9b41834bd8b207156cd5f0633c3f9431cc15ff68aebfb1659c17b`.
  JSON tersebut juga merekam hash seluruh input run/qrels/questions serta kode
  evaluator yang menghasilkannya.
- Manifest pembangunan index contextual:
  `experiments/results/indolaw_200_contextual_index_manifest.json`
- Manifest retrieval contextual:
  `experiments/results/indolaw_200_contextual_retrieval_manifest.json`
- Run Fixed + independent:
  `experiments/results/indolaw_200_corpus_fixed_w150_o30_top50.jsonl`
- Run Fixed + contextual:
  `experiments/results/indolaw_200_corpus_fixed_w150_o30_contextual_e5_top50.jsonl`
- Run SAC + independent:
  `experiments/results/indolaw_200_corpus_sac_w150_o30_s0_top50.jsonl`
- Run SAC + contextual:
  `experiments/results/indolaw_200_corpus_sac_w150_o30_contextual_e5_top50.jsonl`
- Pertanyaan dan qrels:
  `data/evaluation/indolaw_200/corpus_questions.csv`,
  `data/evaluation/indolaw_200/corpus_fixed_w150_o30_qrels.csv`, dan
  `data/evaluation/indolaw_200/corpus_sac_w150_o30_s0_qrels.csv`.
