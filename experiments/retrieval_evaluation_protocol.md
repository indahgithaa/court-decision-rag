# Protokol evaluasi retrieval

Schema metrik aktif: `retrieval-v2-evidence-recall`.

Recall@K dihitung atas unit evidence yang sama untuk kedua strategi. Hasil lama
yang memakai denominator ID chunk per strategi berstatus legacy dan harus
dijalankan ulang sebelum dipakai sebagai hasil final.

## Landasan metodologis

Desain ini mengadaptasi, tetapi tidak menyamakan, beberapa hasil penelitian:

- Sonowal dan Sadhu (2025), [*Structure-Aware Chunking for Abstractive
  Summarization of Long Legal
  Documents*](https://aclanthology.org/2025.justnlp-main.19/), memperkenalkan
  SAC untuk **summarization** putusan hukum panjang. Karena outcome paper
  tersebut bukan retrieval QA, efektivitas SAC untuk repo ini diperlakukan
  sebagai hipotesis yang harus diuji, bukan sebagai asumsi.
- Karpukhin et al. (2020), [*Dense Passage Retrieval for Open-Domain Question
  Answering*](https://aclanthology.org/2020.emnlp-main.550/), memisahkan tahap
  passage retrieval dari reader. Repo ini karena itu mengevaluasi retrieval
  sebelum menambahkan generation agar sumber error dapat diatribusikan.
- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks*](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html),
  menjadi dasar arsitektur retriever–generator, tetapi metrik jawaban tidak
  digunakan untuk menutupi kegagalan retrieval.
- Thakur et al. (2021), [*BEIR: A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information
  Retrieval Models*](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html),
  memotivasi pelaporan metrik ranking terstandardisasi dan evaluasi lintas tipe
  query. Repo ini melaporkan Recall@K, MRR, dan NDCG@K sesuai rancangan skripsi.

Konsekuensinya, variabel yang boleh berubah pada eksperimen utama hanya strategi
chunking. Corpus dokumen, pertanyaan, model embedding, fungsi similarity, dan
nilai `k` harus identik untuk kedua kondisi.

## Tujuan

Evaluasi membandingkan dua kondisi dengan pertanyaan, corpus dokumen, model
embedding, dan parameter pencarian yang sama:

- `fixed_size`: fixed-size dengan overlap 50 kata sebagai baseline;
- `structure_aware`: SAC-H+ dengan batas retoris dan overlap dua kalimat.

Unit analisis utama adalah pertanyaan, sehingga selisih skor kedua strategi
dapat dihitung secara berpasangan.

Sampel `exploration_20` dipakai sebagai pilot untuk memperbaiki pipeline dan
pedoman anotasi. Hasil dari sampel ini tidak menjadi estimasi final karena
dokumennya sudah dipakai ketika mengembangkan detector. Evaluasi final harus
memakai dokumen holdout yang belum digunakan untuk menyusun pola atau memilih
parameter.

## Komposisi pilot

Buat 80 pertanyaan dari 20 dokumen, masing-masing empat pertanyaan:

- satu pertanyaan administratif, bergantian antara `identitas_terdakwa` dan
  `riwayat_penahanan`, sehingga setiap kategori memiliki 10 pertanyaan;
- satu pertanyaan `fakta`;
- satu pertanyaan `pertimbangan_hukum`; dan
- satu pertanyaan `amar_putusan`.

Pertanyaan harus dapat dijawab dari satu dokumen, menggunakan bahasa yang wajar,
dan tidak menyalin kalimat bukti secara utuh. Hindari petunjuk yang hanya cocok
dengan kata-kata persis pada dokumen. Setiap pertanyaan mempunyai satu jawaban
rujukan dan satu rentang bukti minimal yang tetap cukup untuk mendukung jawaban.

Template pilot dibuat dengan:

```powershell
python scripts/prepare_retrieval_questions.py
```

Untuk mempercepat first pass tanpa menganggap keluaran mesin sebagai gold data,
buat draft evidence-grounded dan paket review dengan:

```powershell
python scripts/draft_retrieval_questions.py
```

Script ini selalu memberi status `draft`, menyimpan exact character offsets,
dan menulis konteks bukti ke
`experiments/results/exploration_20_question_review.md`. Reviewer manusia tetap
harus memperbaiki pertanyaan atau bukti yang ambigu sebelum mengubah status
menjadi `approved`.

Periksa progres dan konsistensi baris yang telah diisi dengan:

```powershell
python scripts/validate_retrieval_questions.py
```

## Skema anotasi

Setiap baris memiliki kolom berikut:

| Kolom | Isi |
|---|---|
| `query_id` | ID stabil dan unik. |
| `document_id` | Dokumen yang memuat jawaban. |
| `target_section_label` | Strata pertanyaan yang direncanakan. |
| `question` | Pertanyaan retrieval. |
| `reference_answer` | Jawaban singkat yang didukung dokumen. |
| `evidence_start_position` | Offset awal bukti pada teks bersih dokumen. |
| `evidence_end_position` | Offset akhir eksklusif bukti. |
| `difficulty` | `easy`, `medium`, atau `hard`. |
| `review_status` | `draft`, `approved`, atau `rejected`. |
| `notes` | Catatan ambiguitas atau keputusan reviewer. |

Offset bukti selalu mengacu pada penggabungan `clean_text` halaman menurut
`page_number`, sama seperti input detector dan chunker. Rentang harus diverifikasi
dengan memastikan potongan teks pada offset tersebut sama dengan bukti yang
dipilih.

## Relevansi chunk

Ground truth disimpan sebagai rentang bukti dokumen, bukan sebagai ID chunk.
Setiap unit jawaban memiliki `evidence_id` stabil. Semua chunk Fixed maupun SAC
yang mendukung unit yang sama dipetakan ke `evidence_id` tersebut. Beberapa
kemunculan pasal ekuivalen untuk satu pertanyaan bukan evidence atom terpisah;
semuanya menjadi alternatif dukungan bagi `evidence_id` primer yang sama.
Setelah kedua strategi selesai menghasilkan chunk, kandidat relevansi diturunkan
secara terpisah untuk setiap strategi:

- grade 2: chunk memuat seluruh rentang bukti;
- grade 1: chunk memuat sebagian bukti dan masih cukup untuk menjawab;
- grade 0: chunk tidak cukup untuk menjawab.

Kasus grade 1 harus diperiksa manual karena overlap karakter saja belum menjamin
bahwa konteks jawaban tersedia. Cara ini mencegah skema anotasi menguntungkan
salah satu strategi dan memungkinkan nDCG memakai tingkat relevansi.

Recall@K adalah proporsi `evidence_id` unik yang dicakup oleh sedikitnya satu
chunk pada K hasil teratas. Beberapa overlapping chunk untuk evidence yang sama
hanya dihitung satu kali. Dengan demikian, ukuran dan overlap chunk tidak
mengubah denominator Recall. MRR@K tetap memakai rank chunk relevan pertama.
NDCG@K memakai grade relevansi chunk 0/1/2 dengan gain `2^grade - 1` dan diskon
`log2(rank + 1)`; keterbatasan duplikasi evidence pada overlapping chunk harus
dilaporkan dan dianalisis dalam sensitivity check.

Setelah pertanyaan berstatus `approved`, buat kandidat qrels untuk kedua strategi:

```powershell
python scripts/prepare_retrieval_qrels.py
```

Grade 2 diisi otomatis ketika satu chunk memuat seluruh bukti. Periksa kandidat
grade 1 dan isi `relevance_grade` dengan `1` bila konteksnya cukup untuk menjawab,
atau `0` bila tidak cukup.

## Prosedur eksperimen

1. Selesaikan dan review pertanyaan tanpa melihat hasil retrieval.
2. Bekukan preprocessing, detector, ukuran chunk, overlap, model embedding, dan
   parameter retrieval sebelum menjalankan holdout.
3. Bangun dua index dari corpus yang sama, satu index per strategi.
4. Jalankan setiap pertanyaan pada kedua index dengan nilai `k` yang sama.
5. Laporkan Recall@1/3/5/10, MRR@1/3/5/10, dan NDCG@1/3/5/10 secara keseluruhan
   serta per `target_section_label`.
6. Hitung selisih skor per pertanyaan antara SAC dan fixed-size. Karena empat
   pertanyaan dari putusan yang sama tidak independen, laporkan interval
   kepercayaan dengan *paired cluster bootstrap*: resample dokumen, lalu bawa
   seluruh pertanyaan dokumen terpilih ke setiap replikasi.
7. Tinjau contoh kemenangan dan kegagalan kedua strategi, khususnya bukti yang
   berada dekat batas section atau batas chunk.

## Benchmark eksternal Indo-Law 200

Benchmark robustness memakai 200 dokumen XML ternormalisasi dari Indo-Law.
Split dilakukan berdasarkan pengadilan: 40 dokumen development untuk memilih
desain dan 160 dokumen holdout untuk satu evaluasi final. Grid development
menguji ukuran 150, 300, dan 500 kata dengan overlap 20% pada kedua keluarga.
Aturan seleksi adalah NDCG@5, lalu MRR@5, Recall@5, dan jumlah chunk yang lebih
kecil. Desain lama `fixed_w500_o100` dan `sac_w150_o30_s0` dipilih dengan
denominator Recall per chunk dan kini berstatus `requires_rerun`; desain aktif
baru boleh dibekukan setelah development dijalankan dengan schema evidence.
Selection hasil rerun disimpan sebagai
`experiments/indolaw_200_selected_design_v2.json` agar selection legacy tidak
ditimpa.

Setiap dokumen memiliki empat pertanyaan deterministik berbasis span. Untuk
pertanyaan ketentuan pidana, setiap kemunculan pasal ekuivalen dalam section
pertimbangan hukum diperlakukan relevan; hal ini mencegah evaluasi menghukum
retrieval yang menemukan penyebutan ekuivalen selain span pertama. Karena
section berasal dari anotasi corpus, hasil benchmark ini diberi label
oracle-structure dan dilaporkan terpisah dari evaluasi PDF.

## Diagnosis document-conditioned

Jika analisis error menunjukkan dokumen relevan sudah masuk hasil teratas tetapi
chunk bukti belum ditemukan, jalankan retrieval diagnostik dengan
`--oracle-document-filter`. Pencarian ini memakai `document_id` anotasi untuk
membatasi kandidat sebelum ranking. Tujuannya hanya mengisolasi kualitas ranking
chunk di dalam dokumen dari kualitas pemilihan dokumen.

Hasil ini harus dilaporkan sebagai **gold-document oracle**, tidak boleh
dicampur dengan hasil corpus-wide, dan tidak boleh menjadi dasar klaim performa
sistem end-to-end. Perbandingan tetap memakai pertanyaan, model embedding,
nilai `k`, dan filter dokumen yang sama untuk kedua strategi.

## Sensitivitas reranking

Setelah evaluasi dense retrieval, rerank kandidat top-50 dengan BM25 dan weighted
reciprocal-rank fusion. Gunakan parameter BM25, konstanta RRF, candidate depth,
serta bobot dense yang identik untuk kedua strategi. Sweep bobot dilaksanakan
pada `exploration_20`; semua titik harus dilaporkan agar pemilihan konfigurasi
tidak menyembunyikan hasil yang berlawanan.

Operating point pilot dipilih berdasarkan rata-rata NDCG@5 kedua strategi,
kemudian rata-rata MRR@5, Recall@5, lalu bobot dense yang lebih besar sebagai
tie-breaker.
Bobot terpilih dibekukan sebelum holdout dan tidak boleh dituning ulang dari
hasil holdout. Candidate Recall@50 dilaporkan sebagai coverage: reranker tidak
dapat memulihkan chunk relevan yang tidak masuk kandidat dense awal.

Jika konfigurasi diubah setelah melihat hasil pilot, catat perubahan dan
alasannya. Konfigurasi tersebut kemudian dibekukan sebelum evaluasi holdout.

## Perencanaan ukuran holdout

Sebelum mengumpulkan holdout, jalankan `scripts/10_plan_holdout.py`. Script ini
mengestimasi simpangan baku selisih Recall@5 berpasangan dari pilot, lalu
menghitung kebutuhan pasangan untuk beberapa minimum detectable effect (MDE).
Karena unit sampling adalah
dokumen dan terdapat empat pertanyaan per dokumen, tabel sensitivitas juga
menginflasi kebutuhan dengan design effect untuk beberapa asumsi intraclass
correlation (ICC). MDE, power, dan target dokumen harus dibekukan sebelum hasil
holdout dibuka.

## Kandidat contextual SAC post-hoc

`embedding_text` boleh memuat konteks hierarkis yang diturunkan deterministik
dari section/document, sedangkan `text` dan offset evidence wajib tetap identik
dengan sumber. Kandidat `section_reasoning_document` menambahkan label section
pada semua embedding dan identitas terdakwa hanya pada section analitis. Ia
ditemukan setelah diagnosis hasil lama, sehingga statusnya development post-hoc,
bukan bagian dari frozen design v2. Evaluasi konfirmatori memerlukan holdout baru
atau dataset eksternal yang belum dibuka.

## Kriteria kesiapan

Eksperimen embedding dimulai setelah `validate_sac_compliance.py` lulus dan
seluruh pertanyaan pilot berstatus `approved`. Review struktur lengkap bersifat
diagnostik dan tidak lagi menghambat pembangunan index. Klaim utama baru dibuat
dari holdout yang terpisah pada tingkat dokumen. Pertanyaan dari satu dokumen
tidak boleh dibagi ke set pengembangan dan holdout.
