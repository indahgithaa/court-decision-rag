# Audit penelitian chunking putusan pengadilan

Tanggal audit: 7 Oktober 2026  
Commit yang diaudit: `62f55d60aac0d82308c10876065e3de3b13d1f40` (`main`)  
Draft yang diaudit: `Draft Skripsi - Putu Indah Githa Cahyani - 235150200111041 (1).pdf`  
Status keseluruhan: **PARTIAL - eksperimen retrieval hybrid selesai secara
eksploratif; evaluasi end-to-end/PDF belum selesai**

## Ringkasan eksekutif

**Pembaruan 8 Oktober 2026 (keputusan aktif):** corpus Indo-Law 200 tidak lagi
dibagi menjadi development/holdout untuk analisis skripsi. Fixed-size,
structure-aware, dan hybrid dievaluasi pada satu indeks berisi 200 dokumen dan
800 pertanyaan. Hybrid `hybrid_hier_w150_o30_s0` memperoleh Recall@5 0,6613,
MRR@5 0,4510, nDCG@5 0,4522, dan DRM@5 0,0262. Dibanding fixed 150/30,
peningkatan MRR/nDCG dan penurunan DRM memiliki CI 95% yang tidak melintasi nol;
peningkatan Recall belum demikian. Hybrid tidak unggul pada setiap bagian,
sehingga klaim dominasi per-section tidak diperbolehkan. Karena desain telah
dikembangkan dengan corpus yang sama, status hasil adalah **evaluasi komparatif
eksploratif**, bukan performa test-set independen. Detail ada di
`experiments/hybrid_structure_summary_design.md`.

Semua rujukan development/holdout pada bagian historis dokumen audit ini
mendeskripsikan eksperimen lama dan **bukan desain skripsi aktif**.

Repository sudah memiliki implementasi dasar ekstraksi PDF, pembersihan teks,
deteksi struktur, fixed-size chunking, structure-aware chunking (SAC), dense
retrieval, tiga metrik retrieval resmi, paired cluster bootstrap, dan persiapan
input generasi yang menyamakan anggaran konteks. Desain aktif Indo-Law 200
memakai seluruh dokumen sebagai satu corpus evaluasi.

Namun, angka retrieval tidak dapat dianggap sebagai hasil final skripsi.
Seluruh 200 XML telah direstorasi dari commit sumber terpin dan diverifikasi
terhadap SHA-256 manifest. Evaluasi corpus tunggal telah direproduksi dengan raw
qrels, rankings, index, laporan, dan run manifest lokal. Hasil tetap eksploratif
karena desain dikembangkan secara iteratif pada corpus yang sama. Selain itu,
eksperimen memakai XML ternormalisasi dengan section bawaan, sedangkan
draft mengklaim pipeline utama dimulai dari PDF dan melakukan identifikasi
struktur otomatis. Hasil Indo-Law hanya boleh disebut **oracle-structure
robustness experiment**.

Audit menemukan bahwa Recall@K lama memakai denominator jumlah chunk relevan
masing-masing strategi. Karena jumlah dan granularitas chunk dapat berbeda akibat
ukuran serta overlap, skor lama belum tentu membandingkan unit relevansi yang
sama. Evaluator aktif telah dipindahkan ke schema
`retrieval-v2-evidence-recall`: denominator kini berupa `evidence_id` bersama
dan overlapping chunk untuk bukti yang sama hanya dihitung sekali. Development
selection dan holdout lama telah dijalankan ulang dengan schema tersebut.

Evaluasi generasi belum diimplementasikan. `scripts/06_run_rag.py` masih berhenti
dengan `Not implemented`, sedangkan `src/evaluation/answer_metrics.py` hanya
placeholder. Model generatif, model evaluator Ragas, konfigurasi BERTScore,
retry/failure policy, dan artefak latency end-to-end juga belum ada.

## A. Research status

| Komponen | Status | Bukti dan alasan |
|---|---|---|
| Rumusan masalah dan tujuan | DONE | Draft menetapkan penerapan dan perbandingan SAC dengan chunking konvensional pada performa retrieval dan kualitas jawaban. |
| Corpus Indo-Law 200 dan manifest | DONE (reproduksi lokal) | Seluruh 200 XML direstorasi dari commit sumber terpin; byte dan SHA-256 semuanya cocok. Pipeline aktif menggabungkannya menjadi satu collection `corpus`. |
| Corpus PDF utama | NOT STARTED | Direktori data hanya berisi `.gitkeep`; tidak ada corpus PDF, manifest final, atau split PDF holdout. |
| QA dataset Indo-Law | PARTIAL (review manusia belum selesai) | Benchmark aktif berisi 800 pertanyaan pada satu corpus dengan label `approved` otomatis. Evidence span dan `evidence_id` tersedia, tetapi validasi manusia tetap diperlukan. |
| QA dataset PDF | NOT STARTED | Tidak ada pertanyaan, gold answer, evidence span, atau qrels final untuk corpus PDF. |
| Pembagian corpus Indo-Law | NOT APPLICABLE | Desain skripsi aktif sengaja memakai satu corpus 200 dokumen tanpa development/holdout. |
| Split development/holdout PDF | NOT STARTED | Protokol ada, tetapi corpus dan artefaknya tidak tersedia. |
| Ekstraksi PDF | PARTIAL | Implementasi PyMuPDF dan test sintetis tersedia; belum divalidasi pada corpus final dan tidak ada hasil ekstraksi. |
| Pembersihan teks | PARTIAL | Implementasi dan test pola boilerplate tersedia. Draft menyebut lowercasing serta penghapusan gambar/tabel/garis; implementasi tidak melakukan lowercasing dan hanya bekerja pada teks hasil ekstraksi. Perbedaan perlu diselaraskan. |
| Deteksi struktur otomatis | PARTIAL | Detector regex, fallback `unknown`, preservasi karakter, dan test sintetis tersedia. Belum ada hasil audit manusia pada holdout PDF yang tidak dipakai mengembangkan regex. |
| Fixed-size chunking | DONE | Implementasi word-window 150 kata dan overlap 30 kata tersedia, diuji unit, dan dijalankan pada corpus 200 dokumen. |
| SAC | DONE | Implementasi 150/30 menjaga batas section, tidak membuat chunk lintas bagian, dan dijalankan pada corpus yang sama dengan fixed-size. |
| Hybrid structure-summary | DONE (eksploratif) | Konteks dokumen dan konteks bagian hierarkis ditambahkan hanya pada input embedding; teks/offset bukti tetap verbatim. |
| Embedding dan index | DONE (Indo-Law) / PARTIAL (environment) | Tiga indeks corpus tunggal dibangun dengan `intfloat/multilingual-e5-small`; manifest menyimpan model, hash, commit, dan platform. Lockfile final belum ada. |
| Retrieval komparatif | DONE (eksploratif Indo-Law) / NOT EXTERNAL VALIDATION | Fixed, SAC, dan hybrid dijalankan pada satu indeks masing-masing yang mencakup 200 dokumen. Hasil memakai gold XML section dan tidak boleh disebut evaluasi PDF atau generalisasi data baru. |
| Recall@K | DONE | Schema aktif menghitung coverage `evidence_id` bersama dan mendeduplikasi overlapping chunk; seluruh 800 kueri telah dievaluasi pada satu corpus. |
| MRR@K | DONE (kode dan rerun eksploratif) | Reciprocal rank untuk item relevan pertama diuji termasuk rank K; hasil per-query dan agregat tersedia lokal. |
| NDCG@K | DONE (kode dan rerun eksploratif) | Gain `2^rel-1`, diskon log2, ideal ranking, dan test manual lulus; hasil per-query dan agregat tersedia lokal. |
| Hit@K removal | DONE untuk tree aktif | Tidak ada kecocokan `Hit@K`, `hit_at_k`, `hitk`, atau hit-rate di working tree aktif. Riwayat Git lama masih memuat Hit@K dan diperlakukan sebagai legacy, bukan pipeline aktif. |
| Context budget fairness | PARTIAL | Ketiga chunker memakai 150/30. Audit hybrid menunjukkan maksimum 379 token, P95 279, dan nol input melampaui batas 512; audit generasi jawaban belum dilakukan. |
| Prompt generasi | PARTIAL | `legal_qa_id_v1` tersedia dan sama untuk kedua lengan; belum digunakan dalam eksperimen generasi nyata. |
| Generasi jawaban | NOT STARTED | Tidak ada provider adapter/CLI aktif, model final, output jawaban, retry policy, seed, concurrency, atau failure log. |
| Faithfulness | NOT STARTED | Belum diimplementasikan. |
| Answer Relevance | NOT STARTED | Belum diimplementasikan. |
| BERTScore | NOT STARTED | Dependency opsional tercantum, tetapi encoder/language/rescaling dan metrik utama belum dibekukan. |
| Retrieval latency | PARTIAL | Raw timing tersedia, tetapi merupakan embedding query batch yang diamortisasi plus search. Nilai itu hanya diagnostik; belum ada pengukuran online berulang yang mengendalikan warm-up/noise. |
| Context construction latency | NOT STARTED | Tidak direkam oleh input-preparation script. |
| Generation latency | NOT STARTED | Fungsi timing tersedia, tetapi belum ada runner atau hasil. |
| End-to-end latency | NOT STARTED | Belum dihitung. |
| Statistical uncertainty | PARTIAL | Paired cluster bootstrap pada unit dokumen tersedia. Belum ada effect size, sensitivity analysis final, atau uji alternatif yang dibenarkan oleh asumsi. |
| Qualitative error analysis | PARTIAL | Infrastruktur taxonomy tersedia, tetapi tidak ada artefak final yang dapat diperiksa atau tabel kasus thesis-ready. |
| Ablation | PARTIAL | Grid fixed, Pure SAC lama, dan Pure SAC v2 tersedia pada pasangan 150/30, 300/60, dan 500/100. Boundary-aligned v2 hanya memberi peningkatan kecil pada reasoning 300/60; tidak ada ablation detector/fallback pada PDF. |
| Reproducibility | PARTIAL | Indo-Law dapat direstorasi dan direrun; manifest mencatat hash data/chunks/runs/source, versi package, model, seed, platform, dan commit. Artefak besar masih diabaikan Git, run memakai working tree kotor, dan belum ada lockfile/registry persisten. |
| Thesis-ready Bab IV/V | NOT STARTED | Draft berhenti pada awal Bab IV dan tidak memuat hasil final. |

## Kesesuaian skripsi dan implementasi

| Area | Skripsi mengatakan | Implementasi sekarang | Konsisten? | Tindakan |
|---|---|---|---|---|
| Sumber dan jumlah data | 500 dokumen putusan dari Direktori Putusan MA; rancangan dimulai dari PDF. | Benchmark bernilai angka memakai 200 XML Indo-Law; corpus PDF tidak tersedia. | Tidak | Revisi klaim/corpus draft atau bangun primary PDF experiment. Pertahankan Indo-Law hanya sebagai robustness oracle-structure. |
| Preprocessing | Ekstraksi PDF, normalisasi termasuk lowercasing, penghapusan header/footer, nomor halaman, gambar/tabel/garis/simbol. | PyMuPDF mengekstrak teks horizontal; cleaner menjaga kapitalisasi dan menghapus pola boilerplate konservatif. | Sebagian | Pilih deskripsi yang sesuai implementasi defensible. Jangan klaim penghapusan objek visual oleh cleaner atau lowercasing jika tidak dilakukan. |
| Identifikasi struktur | Penanda struktural dideteksi otomatis dari teks PDF dan ada fallback. | Detector regex otomatis dan fallback `unknown` tersedia untuk pipeline PDF; benchmark hasil memakai label section XML dari corpus. | Sebagian | Pisahkan `automatic PDF structure detection` dari `oracle XML sections` di seluruh tabel dan klaim. |
| Fixed-size baseline | Chunking konvensional sebagai pembanding pada kondisi lain yang sama. | Baseline mengabaikan section. Konfigurasi YAML 300/50, desain Indo-Law terpilih 500/100. | Sebagian | Buat config final yang eksplisit per eksperimen; jangan mengandalkan hardcode dan YAML yang bertentangan. |
| SAC | Memakai struktur logis/hierarkis dokumen. | Section-aware windows; pada XML memakai section gold, pada PDF memakai detector regex. | Ya, dengan batasan | Laporkan dua kondisi secara terpisah dan validasi detector pada holdout PDF. |
| Embedding/vector DB | Draft mencantumkan Sentence Transformers dan Qdrant. | `intfloat/multilingual-e5-small` dan vector store NumPy lokal; tidak ada Qdrant. | Tidak | Ubah draft agar sesuai implementasi atau implementasikan Qdrant tanpa mengubah treatment. Untuk eksperimen final, backend harus identik antarlengan. |
| Retrieval metrics | Recall@K, MRR, NDCG@K. | Ketiganya aktif; Recall memakai evidence unit bersama dan test manual mencakup batas K/graded relevance. | Ya | Pertahankan schema v2; tambahkan sensitivity nDCG berbasis evidence. |
| Statistical analysis | Analisis deskriptif. | Paired cluster bootstrap sudah lebih kuat dan sesuai clustering pertanyaan dalam dokumen. | Tidak, tetapi implementasi lebih defensible | Perbarui Bab III secara transparan untuk memasukkan estimand, unit bootstrap, CI, dan keterbatasan. |
| Kualitas jawaban | Faithfulness, Answer Relevance, BERTScore. | Belum diimplementasikan. | Tidak | Implementasikan runner/evaluator setelah retrieval valid dan konfigurasi model/evaluator dibekukan pada development. |
| Latency | Waktu dari pertanyaan sampai jawaban pada kondisi sama. | Retrieval latency saja tersedia; generator timing belum digunakan. | Tidak | Pisahkan retrieval, context construction, generation, dan end-to-end; tambahkan warm-up/repetisi serta mean/median/p95. |
| Peralatan | Draft mencantumkan versi Python/library dan Qdrant tertentu. | `pyproject.toml` memakai rentang versi, Sentence Transformers `<6`, tanpa Qdrant/Pandas/scikit-learn/TQDM. Audit memasang Python 3.12.10 dan dependency minimal, tetapi belum ada lockfile eksperimen. | Tidak | Buat lockfile/environment capture dari environment eksperimen nyata dan selaraskan daftar perangkat lunak. |
| Fairness prompt/context | Konfigurasi lain harus sama. | Prompt sama, label section tidak dimasukkan, dan word budget sama. | Ya untuk preparasi | Audit token count, tetapkan budget hanya dari development, lalu freeze. |

## B. Critical issues

### Critical

1. **Eksperimen utama dan klaim data tidak cocok.** Draft menyatakan 500 PDF,
   sedangkan satu-satunya hasil tercatat berasal dari 200 XML yang sudah memiliki
   section. Dampak: hasil tidak mendukung klaim ketahanan ekstraksi PDF atau
   deteksi struktur otomatis. Perbaikan: jadikan PDF holdout sebagai eksperimen
   utama dan Indo-Law sebagai robustness oracle-structure, atau revisi ruang
   lingkup skripsi secara eksplisit. Hasil Indo-Law lama masih boleh dipakai hanya
   dengan label oracle-structure dan bukan sebagai hasil utama.

2. **Holdout sudah terkontaminasi oleh evolusi metodologi.** Riwayat Git
   menunjukkan holdout lebih dahulu dilaporkan dengan Hit@5 dan
   `fixed_w300_o60`; setelah itu metrik diganti dan baseline dipilih ulang menjadi
   `fixed_w500_o100`, lalu holdout yang sama dianalisis kembali. Dampak: hasil
   bukan konfirmasi buta. Perbaikan: buat holdout baru yang tidak pernah dibuka
   setelah evaluator/config final dibekukan. Hasil lama dapat dipertahankan
   sebagai corrected exploratory analysis.

3. **Reproduksi lokal selesai, tetapi persistensi artefak belum aman.** Data,
   qrels, chunks, index, raw rankings, laporan, dan manifest telah dibangun ulang
   serta di-hash. Namun `data/`, `vector_db/`, dan `experiments/results/`
   diabaikan Git, sehingga checkout baru masih tidak membawa artefak besar.
   Perbaikan: simpan bundle pada registry persisten dengan checksum dan petunjuk
   restore; jangan mengandalkan workstation ini sebagai satu-satunya salinan.

4. **Unit Recall@K lama tidak fair antarlengan (kode dan rerun v2 selesai).**
   Evaluator lama menghitung proporsi ID chunk relevan, sementara qrels
   dan jumlah chunk relevan dibuat terpisah untuk tiap strategi. Dampak:
   ukuran/overlap dapat mengubah denominator tanpa perubahan evidence semantik.
   Schema aktif kini memakai `evidence_id` bersama dan menolak qrels lama tanpa
   ID tersebut. Development dan holdout lama sudah direrun; output legacy tetap
   ditandai invalid dan tidak ditimpa.

### High

1. **Gold QA lama belum benar-benar gold.** Pertanyaan Indo-Law lama dibuat
   regex dan otomatis berstatus `approved`. Audit menemukan nilai tempat lahir
   yang dapat bocor menjadi label field serta pertanyaan administratif yang
   hampir selalu memilih tempat lahir, bukan penahanan. Generator v2 sudah
   membuat 188 draft development konservatif dan tidak melakukan auto-approve.
   Perbaikan tersisa: review manusia buta terhadap retrieval, catat annotator
   dan keputusan, serta audit inter-annotator agreement bila memungkinkan.

2. **Generation dan answer evaluation belum ada.** Ini langsung membuat tujuan
   penelitian terkait kualitas jawaban belum terjawab. Perbaikan: selesaikan
   runner provider-neutral, freeze model/prompt/decoding, lalu implementasikan
   Faithfulness, Answer Relevance, dan BERTScore dengan metadata evaluator.

3. **Validasi detector hanya sintetis/development.** Tidak ada audit batas
   section pada PDF holdout yang belum pernah dipakai menulis regex. Dampak:
   klaim struktur otomatis tidak tervalidasi. Perbaikan: anotasi dan nilai
   detector pada sampel holdout terpisah sebelum eksperimen utama.

4. **Konfigurasi tersebar dan tidak sepenuhnya terpin.** Desain final Indo-Law
   di-hardcode pada script/grid dan JSON selection, sedangkan YAML utama tetap
   300/50. Dependency memakai rentang versi tanpa lockfile. Perbaikan: satu
   config final per experiment ID, hash config/data, dan capture environment.

5. **Persistensi provenance belum final.** Run manifest otomatis dan larangan
   overwrite sudah ditambahkan; manifest merekam timestamp, commit/dirty state,
   package/model, platform, seed, serta hash input/output/source. Perbaikan
   tersisa: jalankan dari commit bersih, buat lockfile, dan arsipkan artefak yang
   sekarang masih diabaikan Git.

### Medium

1. NDCG masih menilai grade pada unit chunk, sehingga overlapping chunk yang
   mendukung evidence sama dapat muncul lebih dari sekali dalam ranking. Recall
   sudah mendeduplikasi evidence; sensitivity check NDCG masih diperlukan.
2. Context budget disetarakan dalam kata, tetapi belum diaudit dalam token;
   header `[Chunk n]` dan jumlah chunk dapat membuat overhead prompt berbeda.
3. Latency retrieval merupakan embedding batch diamortisasi plus search dan
   belum memiliki warm-up/repetisi; jangan gunakan sebagai klaim latency online
   atau end-to-end.
4. Recall/CI dilaporkan, tetapi belum ada effect size terstandar atau analisis
   sensitivitas terhadap distribusi pertanyaan per section.
5. Benchmark lama memiliki hanya satu query riwayat penahanan pada development
   dan lima pada holdout karena penahanan dijadikan fallback setelah tempat
   lahir. Per-section result tersebut tidak stabil. Draft v2 memisahkan kedua
   tipe dan memperoleh 40 pertanyaan penahanan development, tetapi belum boleh
   dipakai sebagai gold sebelum review manusia.
6. ~~`README.md` tidak ada dan diabaikan `.gitignore`.~~ **Selesai:** README kini
   terlacak dan memuat setup, urutan reproduksi v2, serta batas klaim.

### Low

1. Docstring `src/embedding/__init__.py` dan `src/retrieval/__init__.py` masih
   menyebut komponen sebagai reserved walaupun implementasi sudah ada.
2. Penamaan `NDCG`/`nDCG` belum konsisten di dokumentasi.
3. Draft belum mendefinisikan cutoff MRR secara konsisten sebagai MRR@K di semua
   bagian.

## Audit Hit@K

- Pencarian case-insensitive di seluruh working tree aktif, termasuk notebook,
  config, Markdown, test, script, dan source, tidak menemukan Hit@K atau nama
  implementasi sejenis.
- Riwayat Git sebelum commit `21de070` masih memuat Hit@K. Riwayat tersebut
  adalah provenance legacy dan tidak perlu/semestinya ditulis ulang.
- Artefak v2 aktif telah diperiksa dan tidak memuat field Hit@K; script output
  menolak overwrite agar rerun tidak diam-diam mengganti hasil.

## Audit data dan provenance Indo-Law

- Manifest: 200 dokumen; 40 development dan 160 holdout.
- Pengelompokan: 71 pengadilan; tidak ada pengadilan yang overlap antar-split;
  maksimum empat dokumen per pengadilan.
- Tidak ada duplicate source path atau SHA-256 dalam manifest.
- Sumber terpin pada commit Indo-Law
  `67340338adbc63021630947d905627eef0d7e43f`.
- Waktu akuisisi tercatat `2026-09-30T11:03:32.359240+00:00`.
- Seluruh 200 file lokal berhasil direstorasi dari commit sumber tersebut; semua
  byte dan SHA-256 cocok dengan manifest.

## Validitas angka retrieval yang direproduksi

Dengan desain development v2 (`fixed_w500_o100` vs `sac_w150_o30_s0`), hasil
holdout lama berhasil direproduksi pada 640 pertanyaan: Recall@5 0,5406 vs
0,6375, MRR@5 0,3574 vs 0,3890, dan nDCG@5 0,3476 vs 0,4380 untuk Fixed vs SAC.
Selisih SAC - Fixed beserta paired cluster bootstrap CI 95% adalah +0,0969
[+0,0484, +0,1437] untuk Recall@5, +0,0316 [-0,0032, +0,0647] untuk MRR@5,
dan +0,0904 [+0,0570, +0,1223] untuk nDCG@5.

MRR dan nDCG sama dengan ringkasan lama sampai empat desimal, yang menguatkan
reproduksi ranking. Recall lama 0,4322 vs 0,6080 tidak dipakai karena denominator
chunk spesifik strategi; Recall v2 memakai satu `evidence_id` bersama per query.
Status hasil tetap **corrected exploratory oracle-structure analysis**, bukan
konfirmasi blind.

Analisis per bagian menunjukkan heterogenitas besar: SAC meningkatkan Recall@5
pada `amar_putusan` (0,7812 vs 0,3000) dan `identitas_terdakwa` (0,9613 vs
0,7355), tetapi menurunkannya pada `pertimbangan_hukum` (0,1625 vs 0,5750).
Klaim keunggulan merata tidak didukung.

### Diagnosis lanjutan `pertimbangan_hukum`

Ablation development post-hoc menemukan bahwa short tail chunk bukan penyebab
utama. Penyebab dominan adalah hilangnya konteks hierarkis: pertanyaan memuat
nama perkara, sementara chunk reasoning lokal sering hanya memuat pasal generik.
Kandidat `section_reasoning_document` menambahkan heading pada embedding semua
chunk dan identitas parent hanya pada section analitis, tanpa mengubah teks atau
offset sumber. Pada development, Recall@5 `pertimbangan_hukum` meningkat dari
0,2250 menjadi 0,5250 dan Recall@5 keseluruhan dari 0,6687 menjadi 0,7312.
Kandidat ini berstatus post-hoc dan wajib divalidasi pada data baru; frozen v2
dan hasil holdout lama tidak diubah.

### Keputusan lama: tetap Pure SAC (superseded)

Untuk menjaga ruang lingkup skripsi, contextual candidate tidak dipilih sebagai
desain utama. Pure SAC v2 kemudian diuji dengan batas window yang disejajarkan
ke marker retoris sumber setelah minimal 70% word budget terisi. Ukuran dan
overlap disamakan dengan fixed pada 150/30, 300/60, dan 500/100; tidak ada
context prefix, query filter, atau reranker.

Pada development 300/60, Recall@5 `pertimbangan_hukum` naik kecil dari 0,5500
menjadi 0,5750 dan Recall@10 dari 0,6750 menjadi 0,7000 dibanding Pure SAC lama.
MRR@5 turun dari 0,1812 menjadi 0,1729, sedangkan fixed 300/60 mencapai
Recall@5 0,8000. Pada seleksi nDCG@5 agregat, Pure SAC lama 150/30 tetap kandidat
terbaik. Pure SAC v2 harus dilaporkan sebagai ablation eksploratif, bukan solusi
final. Rancangan dan tabel lengkap ada di `experiments/pure_sac_v2_design.md`.
Evaluasi menyeluruh seluruh cutoff, section, pasangan ukuran, interval bootstrap,
dan overhead indeks ada di
`experiments/fixed_vs_pure_sac_comprehensive_evaluation.md`.

### Keputusan terbaru: hybrid hierarkis

Pure SAC dipertahankan sebagai baseline, bukan lagi metode final. Eksperimen
development membandingkan fixed, pure structure-aware, summary+fixed, satu
fingerprint global+structure, fingerprint+role tag, dan konteks hierarkis.
Desain hierarkis memisahkan identitas dokumen dari konteks bagian sehingga pasal
pertimbangan tidak ditempelkan ke semua chunk amar/dakwaan.

Pada development, `hybrid_hier_w150_o30_s0` mencapai Recall@5 0,6687, MRR@5
0,4626, nDCG@5 0,4681, dan DRM@5 0,0062. Pada holdout eksploratif 160 dokumen,
nilainya menjadi 0,6594, 0,4484, 0,4486, dan 0,0278. Audit token memastikan
tidak ada input hybrid 150/30 yang melampaui 512 token pada development maupun
holdout. Namun fixed tetap lebih tinggi pada Recall@5 pertimbangan hukum
(0,7375 vs 0,6500), sedangkan pure structure-aware lebih tinggi pada amar dan
dakwaan. Dengan demikian kontribusi yang didukung adalah perbaikan kualitas
ranking agregat dan provenance dokumen, bukan keunggulan universal.

## Risiko validitas

- **Construct validity:** chunk-level Recall mungkin tidak merepresentasikan
  evidence coverage yang setara; pertanyaan template dapat lebih mengukur pola
  XML daripada QA hukum alami.
- **Internal validity:** XML gold sections memberi SAC kondisi oracle yang tidak
  dimiliki sistem PDF otomatis; holdout telah dilihat selama evolusi metode.
- **External validity:** corpus hanya putusan pidana khusus narkotika/psikotropika
  tingkat pengadilan tertentu; hasil tidak dapat digeneralisasi ke semua dokumen
  hukum.
- **Conclusion validity:** subgroup sangat timpang; hanya satu run embedding;
  latency online belum dikontrol dan holdout telah dibuka sebelumnya.
- **Reproducibility:** raw data/run kini tersedia lokal dan di-hash, tetapi
  dependency belum terkunci dan artefak hasil diabaikan Git tanpa registry
  alternatif. Run v2 memakai Python 3.12.10 dan mencatat versi package aktual.

## C. Next actions

Urutan kerja yang langsung dapat dilanjutkan:

1. Audit sensitivity nDCG terhadap duplicate overlapping evidence dan
   pertimbangkan metrik nDCG berbasis evidence agar unit ranking konsisten dengan
   Recall sebelum membuka holdout baru.
2. Arsipkan bundle Indo-Law v2 yang sekarang tersedia lokal ke registry
   persisten; validasi checksum saat restore. Run manifest/provenance dan
   larangan overwrite sudah aktif.
3. Bekukan environment Python dengan lockfile dari `.venv` Python 3.12.10.
   Environment capture otomatis sudah ada, tetapi run v2 masih berasal dari
   working tree kotor dan dependency belum terkunci.
4. Review draft dataset v2 secara buta terhadap hasil retrieval, lalu ekspor
   hanya baris yang benar-benar disetujui. Jangan memperlakukan flag `approved`
   otomatis pada benchmark lama atau status `draft` v2 sebagai gold.
5. Perlakukan 160 dokumen lama sebagai exploratory. Buat holdout baru setelah
   evaluator dan desain dibekukan jika klaim konfirmatori masih diperlukan.
6. Bangun primary PDF dataset/split sesuai klaim skripsi; audit detector secara
   manusia pada PDF holdout dan jalankan Fixed vs SAC dengan preprocessing,
   embedding, top-K, serta budget yang sama.
7. Implementasikan generation runner dan evaluator jawaban setelah retrieval
   valid. Pilih model generatif, model judge, dan budget hanya pada development;
   kemudian freeze sebelum holdout.
8. Buat artefak final per-query, agregat, CI, latency, dan error analysis di
   `experiments/final_results/` tanpa menimpa run lama.
9. Selaraskan Bab III/Bab IV: jumlah/sumber data, backend vector store,
   preprocessing aktual, oracle-vs-automatic structure, statistical analysis,
   model/version, serta batas generalisasi.

## Kondisi untuk memakai hasil v2

- Ringkasan Indo-Law v2 boleh dipakai sebagai hasil eksploratif/robustness;
  raw run, qrels, report, dan provenance sudah direproduksi.
- Ringkasan Indo-Law tidak boleh dipakai sebagai validasi detector PDF.
- Ringkasan dari holdout lama tidak boleh disebut konfirmatori buta.
- Angka v2 boleh masuk tabel yang secara eksplisit berlabel eksploratif dan
  oracle-structure. Angka tidak boleh dipakai sebagai hasil utama/final sebelum
  validasi manusia dan validasi PDF/eksternal yang belum pernah dibuka.

## Verifikasi audit

- Seluruh 36 halaman PDF berhasil diekstraksi. Halaman judul, Bab III, dan
  halaman rancangan Bab IV diperiksa secara visual dari render PyMuPDF.
- Visual mengonfirmasi bahwa Bab IV belum selesai: setelah uraian identifikasi
  penanda struktural, dokumen hanya memuat daftar judul subbab 4.3-4.9. Teks
  merujuk Gambar 4.3, tetapi gambar tersebut belum tercantum sebelum daftar
  referensi.
- `python -m compileall -q src scripts tests` lulus.
- Seluruh test lulus: **89 passed** dengan Python 3.12.10. Test yang memakai
  temporary directory harus dijalankan di luar sandbox filesystem; tidak ada
  kegagalan logika atau assertion.
