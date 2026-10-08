# Desain eksperimen SAC + Contextualized Chunk Embeddings

Judul kerja:

> **Penerapan Contextualized Chunk Embeddings Berbasis Structure-Aware
> Chunking pada Retrieval-Augmented Generation untuk Sistem Question Answering
> Putusan Pengadilan di Indonesia**

## Keputusan metode

Eksperimen memakai desain faktorial 2×2 agar efek chunking dan contextualized
embedding tidak tercampur:

| | Independent/early chunking | Contextual/windowed late chunking |
|---|---|---|
| Fixed-size 150/30 | Fixed-Independent | Fixed-Contextual |
| SAC 150/30 | SAC-Independent | SAC-Contextual |

Fixed-Independent adalah baseline konvensional, SAC-Independent adalah ablation
chunking, Fixed-Contextual mengisolasi manfaat contextualization pada batas
fixed, dan SAC-Contextual adalah metode gabungan yang diusulkan. Tidak ada
hierarchical retrieval dalam eksperimen ini.

Pendekatan perlakuan ini adalah contextualized chunk embedding bergaya *late
chunking*. Di dalam setiap family, teks, ID, urutan, dan offset chunk identik
antara kondisi Independent dan Contextual. Query juga memakai encoder dan
konfigurasi yang sama. Perbandingan adalah pipeline early-chunk embedding
konvensional melawan windowed late-chunk pooling; bukan klaim bahwa satu-satunya
operasi numerik yang berubah hanyalah penambahan token konteks.

## Pertanyaan penelitian

1. Apakah contextualized chunk embeddings meningkatkan retrieval pada batas
   Fixed-size dan SAC?
2. Apakah SAC meningkatkan retrieval pada embedding Independent dan Contextual?
3. Apakah terdapat interaksi antara strategi chunking dan contextualization?

Metrik primer adalah evidence Recall@5. MRR@5 dan nDCG@5 adalah metrik ranking
sekunder. Analisis per bagian retoris tetap dilaporkan karena manfaat konteks
kemungkinan tidak seragam. DRM@5 dipakai sebagai diagnosis provenance dokumen.
Seluruh kontras memakai paired document-cluster bootstrap 10.000 sampel dan
win/tie/loss. Tidak ada asumsi bahwa metode harus unggul pada setiap metrik atau
setiap bagian.

## Konfigurasi yang dibekukan

| Komponen | Nilai |
|---|---|
| Corpus | Indo-Law 200, collection `corpus` |
| Chunking | Fixed-size dan SAC, maksimum 150 kata, overlap 30 kata |
| Encoder | `intfloat/multilingual-e5-small` |
| Context window perlakuan | 512 token, overlap 128 token |
| Retrieval | exact cosine, corpus-wide, top-50 |
| Bootstrap | 10.000 paired cluster bootstrap, unit dokumen, seed 42 |

Jendela 512 token mengikuti kapasitas encoder multilingual yang telah dipakai
pada eksperimen sebelumnya. Dokumen yang lebih panjang dibagi menjadi
macro-window yang saling overlap. Setiap chunk ditempatkan pada window yang
memuat seluruh rentang tokennya dan memberikan konteks kiri-kanan paling
seimbang. Pilihan ini menjaga perubahan eksperimen hanya pada waktu pooling,
bukan sekaligus mengganti encoder.

Evaluasi jawaban yang direncanakan memakai Faithfulness, Answer Relevance, dan
BERTScore-F1 pada keempat arm dengan generator, prompt, decoding, context budget,
judge LLM, dan judge embedding yang identik. Tahap ini baru boleh dijalankan
setelah model generator/judge serta kredensial atau compute dibekukan; model
lokal 0,5B tidak dipakai sebagai pengganti karena tidak memadai untuk klaim
evaluasi legal QA final.

## Validasi implementasi

Indexer contextual menolak data jika:

- dokumen sumber tidak ditemukan;
- offset chunk berada di luar dokumen;
- `chunk.text` tidak persis sama dengan `document[start_position:end_position]`;
- tokenizer bukan fast tokenizer dengan offset mapping;
- satu chunk melebihi kapasitas contextual window; atau
- ada chunk yang tidak berhasil dipool.

Validasi terhadap artefak corpus menemukan 200 dokumen, 11.605 chunk Fixed,
12.206 chunk SAC, dan 0 ketidakcocokan offset-teks pada kedua family. Penentuan
window dilakukan terpisah per family; window identik hanya dideduplikasi pada
forward pass sehingga batas satu strategi tidak mengubah embedding strategi
lain.

## Menjalankan eksperimen

```powershell
# 1. Bangun kedua indeks contextual dengan encoding window yang dideduplikasi.
.\.venv\Scripts\python.exe scripts\24_build_factorial_contextual_indexes.py `
  --offline --device cpu

# 2. Embed seluruh query sekali lalu cari pada kedua indeks contextual.
.\.venv\Scripts\python.exe scripts\26_run_factorial_contextual_retrieval.py `
  --offline --device cpu

# 3. Evaluasi keempat arm dan seluruh kontras faktorial yang dibekukan.
.\.venv\Scripts\python.exe scripts\25_evaluate_factorial_contextual_embeddings.py `
  --fixed-contextual-run experiments\results\indolaw_200_corpus_fixed_w150_o30_contextual_e5_top50.jsonl `
  --sac-contextual-run experiments\results\indolaw_200_corpus_sac_w150_o30_contextual_e5_top50.jsonl `
  --output-json experiments\results\indolaw_200_factorial_contextual_evaluation.json `
  --output-md experiments\results\indolaw_200_factorial_contextual_evaluation.md
```

Model E5 sudah tersedia di environment reproduksi lokal. Index contextual tetap
lebih mahal daripada kontrol karena self-attention dijalankan pada macro-window
dokumen, bukan secara terpisah pada chunk pendek.

## Batas klaim

Corpus ini telah digunakan selama pengembangan metode, sehingga hasilnya adalah
perbandingan eksploratif pada satu corpus. Fixed 150/30 adalah matched control
faktorial, bukan konfigurasi Fixed terbaik dari eksperimen lama. nDCG memakai
qrels chunk-specific sehingga kontras lintas Fixed–SAC harus dibaca bersama
evidence Recall; jumlah chunk relevan dapat berbeda karena batas dan overlap.
Klaim generalisasi memerlukan evaluasi eksternal yang belum dipakai untuk
pengembangan.

## Basis paper

Implementasi mengadaptasi Günther et al. (2024), *Late Chunking: Contextual
Chunk Embeddings Using Long-Context Embedding Models*
([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)). Paper tersebut
menempatkan pemisahan/pooling chunk setelah contextual token encoding. Adaptasi
di proyek ini mempertahankan batas SAC yang sudah ada dan memakai overlapping
macro-window 512 token, sehingga istilah yang paling presisi adalah
**windowed late chunking over SAC boundaries**.
