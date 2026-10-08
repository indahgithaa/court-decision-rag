# Desain eksperimen SAC + Contextualized Chunk Embeddings

Judul kerja:

> **Penerapan Contextualized Chunk Embeddings Berbasis Structure-Aware
> Chunking pada Retrieval-Augmented Generation untuk Sistem Question Answering
> Putusan Pengadilan di Indonesia**

## Keputusan metode

Structure-Aware Chunking (SAC) tetap dipakai untuk menentukan batas chunk.
Kontribusi yang diuji adalah **cara representasi chunk**, bukan perubahan batas
chunk atau penambahan hierarchical retrieval.

- **Kontrol — SAC-Independent:** setiap teks chunk SAC di-encode secara terpisah.
- **Perlakuan — SAC-Contextual:** dokumen sumber di-encode dalam contextual
  window, lalu embedding setiap chunk diperoleh dengan mean pooling token yang
  beririsan dengan offset chunk tersebut.

Pendekatan perlakuan ini adalah contextualized chunk embedding bergaya *late
chunking*. Teks, ID, urutan, dan offset seluruh chunk identik pada kedua kondisi.
Query juga memakai encoder dan konfigurasi yang sama. Karena itu selisih hasil
dapat dikaitkan lebih bersih dengan contextualization pada embedding chunk.

## Pertanyaan penelitian

> Apakah contextualized chunk embeddings meningkatkan kinerja retrieval RAG
> dibanding independent chunk embeddings ketika keduanya menggunakan batas
> Structure-Aware Chunking yang identik pada putusan pengadilan Indonesia?

Hipotesis utama dinilai pada Recall@5, MRR@5, dan nDCG@5. Analisis per bagian
retoris tetap dilaporkan karena manfaat konteks kemungkinan tidak seragam.
DRM@5 dipakai sebagai diagnosis apakah konteks membantu menjaga provenance
dokumen. Tidak ada asumsi bahwa metode harus unggul pada setiap metrik atau
setiap bagian.

## Konfigurasi yang dibekukan

| Komponen | Nilai |
|---|---|
| Corpus | Indo-Law 200, collection `corpus` |
| Chunking | SAC, maksimum 150 kata, overlap 30 kata |
| Encoder | `intfloat/multilingual-e5-small` |
| Context window perlakuan | 512 token, overlap 128 token |
| Retrieval | exact cosine, corpus-wide, top-50 |
| Bootstrap | paired cluster bootstrap, unit dokumen, seed 42 |

Jendela 512 token mengikuti kapasitas encoder multilingual yang telah dipakai
pada eksperimen sebelumnya. Dokumen yang lebih panjang dibagi menjadi
macro-window yang saling overlap. Setiap chunk ditempatkan pada window yang
memuat seluruh rentang tokennya dan memberikan konteks kiri-kanan paling
seimbang. Pilihan ini menjaga perubahan eksperimen hanya pada waktu pooling,
bukan sekaligus mengganti encoder.

## Validasi implementasi

Indexer contextual menolak data jika:

- dokumen sumber tidak ditemukan;
- offset chunk berada di luar dokumen;
- `chunk.text` tidak persis sama dengan `document[start_position:end_position]`;
- tokenizer bukan fast tokenizer dengan offset mapping;
- satu chunk melebihi kapasitas contextual window; atau
- ada chunk yang tidak berhasil dipool.

Validasi terhadap artefak corpus saat implementasi menemukan 200 dokumen,
12.206 chunk SAC, dan 0 ketidakcocokan offset-teks.

## Menjalankan eksperimen

```powershell
# 1. Indeks SAC independen yang lama dipakai sebagai kontrol tervalidasi.
# Bangun hanya indeks contextual dari chunk SAC yang sama.
.\.venv\Scripts\python.exe scripts\04_build_index.py `
  --config configs\indolaw_sac_contextual.yaml

# 2. Jalankan retrieval contextual dengan query dan top-k yang sama.
.\.venv\Scripts\python.exe scripts\05_run_retrieval.py `
  --config configs\indolaw_sac_contextual.yaml `
  --questions data\evaluation\indolaw_200\corpus_questions.csv `
  --output experiments\results\sac_contextual_e5_top50.jsonl

# 3. Hitung metrik, hasil per bagian, dan interval bootstrap berpasangan.
.\.venv\Scripts\python.exe scripts\23_evaluate_contextual_embeddings.py `
  --independent-run experiments\results\indolaw_200_corpus_sac_w150_o30_s0_top50.jsonl `
  --contextual-run experiments\results\sac_contextual_e5_top50.jsonl `
  --output-json experiments\results\sac_contextual_embedding_evaluation.json `
  --output-md experiments\results\sac_contextual_embedding_evaluation.md
```

Model E5 sudah tersedia di environment reproduksi lokal. Index contextual tetap
lebih mahal daripada kontrol karena self-attention dijalankan pada macro-window
dokumen, bukan secara terpisah pada chunk pendek.

## Batas klaim

Corpus ini telah digunakan selama pengembangan metode, sehingga hasilnya adalah
perbandingan eksploratif pada satu corpus. Jika konfigurasi atau ukuran window
dipilih setelah melihat hasil, pilihan tersebut harus dicatat sebagai tuning.
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
