# Protokol evaluasi retrieval

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
  query. Cutoff primer repo ini tetap Hit@5 karena corpus pilot kecil dan jumlah
  relevant chunks berubah akibat strategi overlap.

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
Setelah kedua strategi selesai menghasilkan chunk, kandidat relevansi diturunkan
secara terpisah untuk setiap strategi:

- grade 2: chunk memuat seluruh rentang bukti;
- grade 1: chunk memuat sebagian bukti dan masih cukup untuk menjawab;
- grade 0: chunk tidak cukup untuk menjawab.

Kasus grade 1 harus diperiksa manual karena overlap karakter saja belum menjamin
bahwa konteks jawaban tersedia. Cara ini mencegah skema anotasi menguntungkan
salah satu strategi dan memungkinkan nDCG memakai tingkat relevansi.

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
5. Gunakan Hit@5 sebagai metrik primer karena jumlah chunk relevan dapat berbeda
   akibat overlap. Laporkan juga Hit@1/3/10, MRR@1/3/5/10, nDCG@1/3/5/10, dan
   Recall@1/3/5/10 secara keseluruhan serta per `target_section_label`.
6. Hitung selisih skor per pertanyaan antara SAC dan fixed-size. Laporkan interval
   kepercayaan bootstrap berpasangan untuk selisih rata-rata pada evaluasi final.
7. Tinjau contoh kemenangan dan kegagalan kedua strategi, khususnya bukti yang
   berada dekat batas section atau batas chunk.

Jika konfigurasi diubah setelah melihat hasil pilot, catat perubahan dan
alasannya. Konfigurasi tersebut kemudian dibekukan sebelum evaluasi holdout.

## Kriteria kesiapan

Eksperimen embedding dimulai setelah `validate_sac_compliance.py` lulus dan
seluruh pertanyaan pilot berstatus `approved`. Review struktur lengkap bersifat
diagnostik dan tidak lagi menghambat pembangunan index. Klaim utama baru dibuat
dari holdout yang terpisah pada tingkat dokumen. Pertanyaan dari satu dokumen
tidak boleh dibagi ke set pengembangan dan holdout.
