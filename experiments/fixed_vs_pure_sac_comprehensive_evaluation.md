# Evaluasi komprehensif Fixed-Size vs Pure SAC

Tanggal analisis: 7 Oktober 2026  
Dataset: Indo-Law 200, split development  
Status: eksploratif, *oracle structure*, belum merupakan hasil holdout final

Laporan ini mencakup evaluasi retrieval yang sudah benar-benar dijalankan.
Faithfulness, Answer Relevance, BERTScore, dan latency end-to-end belum tersedia
karena runner generasi jawaban belum diimplementasikan; metrik tersebut tidak
diestimasi atau diisi dengan angka simulasi.

## 1. Ringkasan eksekutif

Evaluasi ini membandingkan sembilan konfigurasi:

- Fixed-size pada 150/30, 300/60, dan 500/100;
- Pure SAC v1 pada 150/30, 300/60, dan 500/100;
- Pure SAC v2 boundary-aligned pada 150/30, 300/60, dan 500/100.

Angka pertama menunjukkan maksimum kata per chunk dan angka kedua menunjukkan
overlap kata. Semua pasangan memakai overlap 20%, model
`intfloat/multilingual-e5-small`, normalisasi embedding, dense retrieval
full-corpus, dan `top_k=50` yang sama.

Kesimpulan utamanya:

1. Tidak ada satu konfigurasi yang menang pada semua metrik dan cutoff.
2. Pure SAC v1 150/30 mempunyai nDCG@5 agregat tertinggi, yaitu 0,4587.
3. Pure SAC v1 300/60 mempunyai MRR@5 tertinggi, yaitu 0,4322.
4. Pure SAC v1 150/30 mempunyai Recall@5 tertinggi, yaitu 0,6687.
5. Fixed 500/100 mempunyai Recall@50 tertinggi, yaitu 0,9187, dengan indeks
   paling kecil, hanya 710 chunk.
6. SAC unggul kuat pada `identitas_terdakwa` dan `amar_putusan`.
7. Fixed unggul kuat pada `pertimbangan_hukum`; hasil terbaiknya adalah fixed
   300/60 dengan Recall@5 0,8000.
8. SAC v2 300/60 memperbaiki Recall@5 reasoning dari 0,5500 menjadi 0,5750
   dibanding SAC v1 pada ukuran sama, tetapi perbaikannya kecil dan tidak
   konsisten pada MRR/nDCG.
9. Hasil `riwayat_penahanan` belum dapat ditafsirkan karena benchmark lama hanya
   memiliki satu kueri development untuk section tersebut.

Jika aturan pemilihan adalah memaksimalkan nDCG@5, dilanjutkan MRR@5,
Recall@5, kemudian meminimalkan jumlah chunk, kandidat per keluarga adalah:

- Fixed-size: `fixed_w500_o100`;
- Pure SAC: `sac_w150_o30_s0`.

Perbandingan dua kandidat keluarga tersebut bukan perbandingan parameter yang
sepenuhnya berpasangan karena ukuran chunk berbeda. Untuk mengisolasi pengaruh
strategi chunking, interpretasi utama tetap harus menggunakan pasangan dengan
ukuran dan overlap sama.

## 2. Desain evaluasi

| Komponen | Nilai |
|---|---|
| Dokumen development | 40 |
| Kueri | 160 |
| Retrieval scope | Seluruh corpus development |
| Model embedding | `intfloat/multilingual-e5-small` |
| Cutoff | 1, 3, 5, 10, 50 |
| Unit Recall | Evidence semantik bersama, bukan jumlah chunk relevan |
| MRR dan nDCG | Dihitung pada ranking chunk |
| Reranker | Tidak ada |
| Contextual embedding | Tidak digunakan pada Pure SAC |
| Bootstrap | Paired cluster bootstrap, unit dokumen, 10.000 sampel, seed 42 |

Distribusi kueri benchmark lama:

| Target section | Jumlah kueri |
|---|---:|
| `amar_putusan` | 40 |
| `identitas_terdakwa` | 39 |
| `pertimbangan_hukum` | 40 |
| `riwayat_dakwaan` | 40 |
| `riwayat_penahanan` | 1 |
| **Total** | **160** |

Ketimpangan ini penting: skor penahanan 0 atau 1 hanya menggambarkan satu
kueri dan tidak dapat digunakan untuk menyimpulkan kualitas umum suatu metode.

## 3. Arti metrik

- **Recall@K**: proporsi evidence yang berhasil ditemukan dalam K hasil teratas.
  Overlapping chunk yang merujuk evidence sama tidak memperbesar denominator.
- **MRR@K**: menilai seberapa awal chunk relevan pertama muncul. Nilai tinggi
  berarti pengguna lebih cepat memperoleh evidence relevan.
- **nDCG@K**: menilai kualitas urutan dan graded relevance pada ranking chunk.
  Metrik ini masih sensitif terhadap beberapa overlapping chunk yang mendukung
  evidence sama; karena itu dibaca bersama Recall dan MRR.

## 4. Hasil agregat semua cutoff

### 4.1 Recall

| Konfigurasi | @1 | @3 | @5 | @10 | @50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,2188 | 0,4500 | 0,6562 | **0,8063** | 0,8938 |
| SAC v1 150/30 | 0,2938 | 0,5125 | **0,6687** | 0,7250 | 0,8125 |
| SAC v2 150/30 | 0,2812 | 0,4813 | 0,6375 | 0,7000 | 0,7688 |
| Fixed 300/60 | 0,2375 | 0,5250 | 0,6562 | 0,7562 | 0,8688 |
| SAC v1 300/60 | **0,3250** | 0,5062 | 0,6500 | 0,7750 | 0,8938 |
| SAC v2 300/60 | 0,3187 | 0,5000 | 0,6625 | 0,7750 | 0,8875 |
| Fixed 500/100 | 0,2812 | **0,6062** | 0,6500 | 0,7438 | **0,9187** |
| SAC v1 500/100 | **0,3250** | 0,5125 | 0,6125 | 0,6875 | 0,8750 |
| SAC v2 500/100 | 0,3187 | 0,4938 | 0,5938 | 0,7000 | 0,8500 |

Interpretasi:

- SAC lebih kuat pada rank pertama, terutama v1 300/60 dan 500/100.
- Fixed 500/100 paling kuat pada Recall@3 dan Recall@50.
- Pure SAC v1 150/30 paling kuat pada cutoff utama Recall@5.
- Pada cutoff lebih dalam, fixed cenderung mengejar atau melampaui SAC.

### 4.2 MRR

| Konfigurasi | @1 | @3 | @5 | @10 | @50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,2188 | 0,3156 | 0,3622 | 0,3821 | 0,3872 |
| SAC v1 150/30 | 0,2938 | 0,3771 | 0,4146 | 0,4229 | 0,4272 |
| SAC v2 150/30 | 0,2812 | 0,3573 | 0,3939 | 0,4027 | 0,4058 |
| Fixed 300/60 | 0,2375 | 0,3604 | 0,3914 | 0,4043 | 0,4103 |
| SAC v1 300/60 | **0,3250** | 0,4000 | **0,4322** | **0,4484** | **0,4542** |
| SAC v2 300/60 | 0,3187 | 0,3937 | 0,4303 | 0,4457 | 0,4508 |
| Fixed 500/100 | 0,2812 | **0,4208** | 0,4315 | 0,4440 | 0,4533 |
| SAC v1 500/100 | **0,3250** | 0,4021 | 0,4255 | 0,4366 | 0,4451 |
| SAC v2 500/100 | 0,3187 | 0,3896 | 0,4127 | 0,4274 | 0,4340 |

Interpretasi:

- SAC v1 300/60 mempunyai MRR@5, @10, dan @50 tertinggi.
- Fixed 500/100 hampir menyamai MRR@5 SAC v1 300/60, dengan selisih 0,0007.
- SAC v2 tidak mengungguli SAC v1 pada MRR untuk ukuran mana pun.

### 4.3 nDCG

| Konfigurasi | @1 | @3 | @5 | @10 | @50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,2188 | 0,2815 | 0,3492 | 0,4035 | 0,4233 |
| SAC v1 150/30 | 0,2938 | 0,4035 | **0,4587** | 0,4748 | 0,4898 |
| SAC v2 150/30 | 0,2812 | 0,3697 | 0,4279 | 0,4501 | 0,4634 |
| Fixed 300/60 | 0,2375 | 0,3319 | 0,3779 | 0,4148 | 0,4484 |
| SAC v1 300/60 | **0,3250** | 0,4084 | 0,4554 | **0,4921** | **0,5186** |
| SAC v2 300/60 | 0,3187 | 0,4068 | 0,4572 | 0,4905 | 0,5142 |
| Fixed 500/100 | 0,2812 | 0,3991 | 0,4168 | 0,4541 | 0,4978 |
| SAC v1 500/100 | **0,3250** | **0,4156** | 0,4441 | 0,4715 | 0,5118 |
| SAC v2 500/100 | 0,3187 | 0,4078 | 0,4410 | 0,4704 | 0,5037 |

Interpretasi:

- SAC mengungguli fixed pada nDCG@5 di semua ukuran yang sama.
- SAC v1 150/30 memberikan nDCG@5 tertinggi.
- SAC v1 300/60 memberikan nDCG@10 dan nDCG@50 tertinggi.
- SAC v2 hanya sedikit melampaui SAC v1 pada nDCG@5 ukuran 300/60
  (0,4572 vs 0,4554), tetapi perbedaan ini sangat kecil.

## 5. Perbandingan statistik pada cutoff utama @5

Delta adalah metode A dikurangi metode B. Interval kepercayaan dihitung dengan
paired cluster bootstrap pada 40 dokumen. Interval yang melintasi nol berarti
data development belum menunjukkan arah perbedaan yang stabil.

Keterangan: F = Fixed, V1 = Pure SAC v1, V2 = Pure SAC v2.

| Ukuran | Perbandingan | Metrik | Delta | CI 95% |
|---:|---|---|---:|---:|
| 150/30 | V1 − F | Recall@5 | +0,0125 | [-0,0625, +0,0938] |
| 150/30 | V1 − F | MRR@5 | +0,0524 | [-0,0160, +0,1241] |
| 150/30 | V1 − F | nDCG@5 | **+0,1095** | **[+0,0570, +0,1651]** |
| 150/30 | V2 − F | Recall@5 | -0,0187 | [-0,0875, +0,0500] |
| 150/30 | V2 − F | MRR@5 | +0,0317 | [-0,0340, +0,0999] |
| 150/30 | V2 − F | nDCG@5 | **+0,0788** | **[+0,0264, +0,1350]** |
| 150/30 | V2 − V1 | Recall@5 | -0,0312 | [-0,0688, 0,0000] |
| 150/30 | V2 − V1 | MRR@5 | **-0,0207** | **[-0,0377, -0,0049]** |
| 150/30 | V2 − V1 | nDCG@5 | **-0,0308** | **[-0,0450, -0,0176]** |
| 300/60 | V1 − F | Recall@5 | -0,0063 | [-0,1062, +0,0938] |
| 300/60 | V1 − F | MRR@5 | +0,0408 | [-0,0313, +0,1099] |
| 300/60 | V1 − F | nDCG@5 | **+0,0775** | **[+0,0123, +0,1426]** |
| 300/60 | V2 − F | Recall@5 | +0,0063 | [-0,0938, +0,1062] |
| 300/60 | V2 − F | MRR@5 | +0,0390 | [-0,0305, +0,1065] |
| 300/60 | V2 − F | nDCG@5 | **+0,0793** | **[+0,0150, +0,1435]** |
| 300/60 | V2 − V1 | Recall@5 | +0,0125 | [-0,0187, +0,0437] |
| 300/60 | V2 − V1 | MRR@5 | -0,0019 | [-0,0132, +0,0090] |
| 300/60 | V2 − V1 | nDCG@5 | +0,0018 | [-0,0090, +0,0132] |
| 500/100 | V1 − F | Recall@5 | -0,0375 | [-0,1437, +0,0688] |
| 500/100 | V1 − F | MRR@5 | -0,0059 | [-0,0853, +0,0791] |
| 500/100 | V1 − F | nDCG@5 | +0,0273 | [-0,0492, +0,1087] |
| 500/100 | V2 − F | Recall@5 | -0,0563 | [-0,1562, +0,0500] |
| 500/100 | V2 − F | MRR@5 | -0,0187 | [-0,0974, +0,0653] |
| 500/100 | V2 − F | nDCG@5 | +0,0243 | [-0,0526, +0,1051] |
| 500/100 | V2 − V1 | Recall@5 | -0,0187 | [-0,0563, +0,0187] |
| 500/100 | V2 − V1 | MRR@5 | **-0,0128** | **[-0,0263, -0,0007]** |
| 500/100 | V2 − V1 | nDCG@5 | -0,0030 | [-0,0131, +0,0085] |

Temuan statistik development:

- Keunggulan nDCG@5 SAC terhadap fixed cukup konsisten pada ukuran 150 dan 300.
- Belum ada perbedaan Recall@5 atau MRR@5 SAC-vs-fixed yang intervalnya tidak
  melintasi nol.
- SAC v2 150/30 secara konsisten lebih buruk daripada SAC v1 untuk MRR@5 dan
  nDCG@5.
- SAC v2 300/60 praktis setara dengan SAC v1 300/60 secara agregat.
- SAC v2 500/100 mempunyai MRR@5 lebih rendah daripada SAC v1.

Analisis ini bersifat eksploratif karena konfigurasi v2 dibuat setelah melihat
kelemahan development. CI tidak mengubahnya menjadi hasil konfirmatori.

## 6. Hasil per section

### 6.1 Amar putusan — 40 kueri

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,7000 | 0,3875 | 0,4448 | 0,8000 | 0,9250 |
| SAC v1 150/30 | 0,8250 | 0,4100 | 0,5128 | 0,9000 | 0,9500 |
| SAC v2 150/30 | 0,7750 | 0,3879 | 0,4271 | 0,9000 | 0,9500 |
| Fixed 300/60 | 0,6250 | 0,4008 | 0,4509 | 0,7250 | 0,8750 |
| SAC v1 300/60 | 0,8500 | 0,5746 | 0,6438 | 0,9250 | 0,9500 |
| SAC v2 300/60 | 0,8750 | 0,5787 | 0,6541 | 0,9250 | 0,9500 |
| Fixed 500/100 | 0,4250 | 0,2604 | 0,2815 | 0,5500 | 0,8000 |
| SAC v1 500/100 | **0,9000** | **0,6050** | **0,6797** | **0,9250** | **0,9500** |
| SAC v2 500/100 | **0,9000** | **0,6050** | **0,6797** | **0,9250** | **0,9500** |

SAC merupakan pilihan jelas untuk amar putusan. Batas section mencegah amar
tercampur dengan reasoning atau metadata penutup. Ukuran 500/100 paling baik
untuk ranking awal, sedangkan SAC v1 dan v2 identik pada ukuran tersebut.

### 6.2 Identitas terdakwa — 39 kueri

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,6410 | 0,3889 | 0,4502 | 0,8718 | **1,0000** |
| SAC v1 150/30 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9487 |
| SAC v2 150/30 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9487 |
| Fixed 300/60 | 0,5897 | 0,3081 | 0,3777 | 0,7436 | 0,8974 |
| SAC v1 300/60 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9744 |
| SAC v2 300/60 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9744 |
| Fixed 500/100 | 0,7949 | 0,6141 | 0,6590 | **0,9231** | **1,0000** |
| SAC v1 500/100 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9744 |
| SAC v2 500/100 | **0,8974** | **0,8974** | **0,8974** | 0,8974 | 0,9744 |

SAC sangat kuat pada identitas karena section identitas pendek, spesifik, dan
langsung mengandung nama serta atribut yang ditanyakan. Hasil semua ukuran SAC
hampir identik. Fixed dapat mengejar pada cutoff besar, tetapi ranking awalnya
lebih lemah.

### 6.3 Pertimbangan hukum — 40 kueri

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,7500 | 0,5079 | 0,2456 | 0,8000 | 0,8250 |
| SAC v1 150/30 | 0,2250 | 0,0958 | 0,0530 | 0,3000 | 0,4750 |
| SAC v2 150/30 | 0,1750 | 0,0483 | 0,0321 | 0,2000 | 0,3000 |
| Fixed 300/60 | **0,8000** | **0,5279** | **0,2991** | **0,8250** | 0,9000 |
| SAC v1 300/60 | 0,5500 | 0,1812 | 0,1498 | 0,6750 | 0,8500 |
| SAC v2 300/60 | 0,5750 | 0,1729 | 0,1494 | 0,7000 | 0,8250 |
| Fixed 500/100 | 0,7000 | 0,3937 | 0,2393 | 0,7750 | **0,9500** |
| SAC v1 500/100 | 0,4500 | 0,1571 | 0,1176 | 0,5250 | 0,8250 |
| SAC v2 500/100 | 0,3750 | 0,1038 | 0,1039 | 0,5500 | 0,7500 |

Fixed unggul pada seluruh ukuran untuk pertimbangan hukum. Penyebab utamanya:

- reasoning merupakan section yang sangat panjang;
- banyak potongan membahas pasal yang generik dan mirip antardokumen;
- pertanyaan menyebut nama terdakwa, tetapi nama itu tidak selalu muncul di
  chunk reasoning lokal;
- fixed dapat melintasi batas section sehingga kadang membawa petunjuk identitas
  atau konteks tetangga yang membantu membedakan dokumen.

Pure SAC v2 300/60 hanya memberi peningkatan coverage kecil dibanding v1.
Menyuntikkan identitas parent akan mengatasi sebagian masalah, tetapi metode itu
menjadi contextual SAC dan berada di luar perbandingan Pure SAC ini.

### 6.4 Riwayat dakwaan — 40 kueri

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 0,5250 | 0,1692 | 0,2577 | 0,7500 | 0,8250 |
| SAC v1 150/30 | **0,7500** | 0,2775 | 0,3940 | **0,8250** | 0,9000 |
| SAC v2 150/30 | 0,7250 | 0,2642 | 0,3776 | **0,8250** | 0,9000 |
| Fixed 300/60 | 0,6250 | 0,3362 | 0,3932 | 0,7500 | 0,8250 |
| SAC v1 300/60 | 0,3250 | 0,0979 | 0,1530 | 0,6250 | 0,8250 |
| SAC v2 300/60 | 0,3250 | 0,0946 | 0,1501 | 0,6000 | 0,8250 |
| Fixed 500/100 | 0,7000 | **0,4729** | **0,5038** | 0,7500 | **0,9500** |
| SAC v1 500/100 | 0,2250 | 0,0650 | 0,1040 | 0,4250 | 0,7750 |
| SAC v2 500/100 | 0,2250 | 0,0671 | 0,1056 | 0,4500 | 0,7500 |

Hasil dakwaan sangat sensitif terhadap ukuran. SAC 150/30 memberi Recall@5
tertinggi, sedangkan fixed 500/100 memberi ranking dan coverage mendalam terbaik.
Chunk SAC yang besar cenderung terlalu banyak memuat uraian dakwaan yang mirip,
sehingga diskriminasi dense embedding menurun.

### 6.5 Riwayat penahanan — satu kueri

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 1,0000 | 0,2000 | 0,3869 | 1,0000 | 1,0000 |
| Delapan konfigurasi lain | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

Angka ini tidak boleh dipakai untuk menyatakan bahwa SAC gagal total pada
penahanan. Benchmark lama hanya mempunyai satu kueri penahanan karena generator
memilih tempat lahir terlebih dahulu dan menjadikan penahanan sebagai fallback.
Dataset v2 sudah memisahkan kedua jenis pertanyaan dan menghasilkan 40 draft
penahanan development, tetapi semuanya masih menunggu review manusia.

## 7. Analisis keberhasilan berpasangan pada Recall@5

Tabel berikut menunjukkan jumlah kueri yang ditemukan atau gagal oleh kedua
metode. A adalah metode di sebelah kiri tanda minus.

| Ukuran | Perbandingan | Keduanya berhasil | Hanya A | Hanya B | Keduanya gagal |
|---:|---|---:|---:|---:|---:|
| 150/30 | V1 − Fixed | 74 | 33 | 31 | 22 |
| 150/30 | V2 − Fixed | 73 | 29 | 32 | 26 |
| 150/30 | V2 − V1 | 101 | 1 | 6 | 52 |
| 300/60 | V1 − Fixed | 74 | 30 | 31 | 25 |
| 300/60 | V2 − Fixed | 76 | 30 | 29 | 25 |
| 300/60 | V2 − V1 | 101 | 5 | 3 | 51 |
| 500/100 | V1 − Fixed | 68 | 30 | 36 | 26 |
| 500/100 | V2 − Fixed | 66 | 29 | 38 | 27 |
| 500/100 | V2 − V1 | 93 | 2 | 5 | 60 |

Fixed dan SAC menemukan himpunan kueri relevan yang tidak sepenuhnya sama.
Contohnya pada 150/30, 33 kueri hanya berhasil ditemukan SAC v1, sedangkan 31
kueri hanya berhasil ditemukan fixed. Ini menjelaskan mengapa rerata Recall
mirip walaupun perilaku per section sangat berbeda.

## 8. Efisiensi dan karakteristik chunk

| Konfigurasi | Chunk | Chunk/dokumen | Mean kata | Median | P95 kata | Qrel positif |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 150/30 | 2.341 | 58,52 | 148,8 | 150 | 150 | 272 |
| SAC v1 150/30 | 2.457 | 61,42 | 138,4 | 150 | 150 | 255 |
| SAC v2 150/30 | 2.526 | 63,15 | 139,1 | 150 | 150 | 262 |
| Fixed 300/60 | 1.173 | 29,32 | 296,0 | 300 | 300 | 266 |
| SAC v1 300/60 | 1.346 | 33,65 | 248,4 | 300 | 300 | 243 |
| SAC v2 300/60 | 1.385 | 34,62 | 252,4 | 300 | 300 | 246 |
| Fixed 500/100 | **710** | **17,75** | 487,7 | 500 | 500 | 251 |
| SAC v1 500/100 | 923 | 23,07 | 356,3 | 500 | 500 | 231 |
| SAC v2 500/100 | 947 | 23,68 | 371,4 | 478 | 500 | 233 |

Dibanding fixed dengan ukuran sama, overhead jumlah chunk adalah:

| Ukuran | SAC v1 vs Fixed | SAC v2 vs Fixed |
|---:|---:|---:|
| 150/30 | +5,0% | +7,9% |
| 300/60 | +14,7% | +18,1% |
| 500/100 | +30,0% | +33,4% |

SAC menghasilkan lebih banyak chunk karena setiap section di-window secara
mandiri dan sisa pendek pada satu section tidak dapat digabungkan dengan section
berikutnya. Overhead meningkat pada ukuran besar karena semakin banyak section
yang panjangnya jauh di bawah maksimum chunk.

Jumlah qrel positif bukan metrik kualitas. Nilainya menunjukkan berapa banyak
chunk yang sepenuhnya mencakup evidence. Recall aktif mendeduplikasi chunk yang
mendukung evidence sama agar strategi dengan lebih banyak overlap tidak mendapat
keuntungan denominator.

Latency raw tidak dibandingkan dalam laporan ini. Run SAC v2 dibuat pada sesi
yang berbeda dari fixed/SAC v1, sementara timing yang tercatat merupakan query
embedding batch yang diamortisasi ditambah search. Karena itu selisih latency
lintas sesi tidak dapat diatribusikan kepada strategi chunking.

### 8.1 Integritas artefak chunk

Pemeriksaan langsung terhadap seluruh 13.808 chunk dari sembilan konfigurasi
menunjukkan:

- nol chunk dengan teks yang berbeda dari substring pada offset dokumen;
- nol chunk yang melebihi `max_words` konfigurasinya;
- nol chunk SAC v1/v2 yang melintasi batas section;
- seluruh section mempertahankan cakupan karakter dan overlap intra-section.

Validator SAC-H+ umum tetap menandai pemeriksaan trigger heading sebagai gagal
karena heading Indo-Law berbentuk nama tag XML (`pertimbangan_hukum` dan
`amar_putusan`), bukan frasa PDF seperti `Menimbang` atau `Mengadili`. Ini bukan
kegagalan offset/chunking, tetapi bukti bahwa eksperimen memakai oracle XML dan
tidak dapat dianggap validasi detector struktur PDF.

## 9. Pemenang menurut tujuan

| Tujuan | Konfigurasi yang paling sesuai | Alasan |
|---|---|---|
| nDCG@5 agregat | SAC v1 150/30 | Nilai tertinggi 0,4587 |
| MRR@5 agregat | SAC v1 300/60 | Nilai tertinggi 0,4322 |
| Recall@5 agregat | SAC v1 150/30 | Nilai tertinggi 0,6687 |
| Recall@50 agregat | Fixed 500/100 | Nilai tertinggi 0,9187 |
| Indeks terkecil | Fixed 500/100 | Hanya 710 chunk |
| Amar putusan | SAC v1/v2 500/100 | Recall@5 0,9000; nDCG@5 0,6797 |
| Identitas terdakwa | Semua SAC | Recall/MRR/nDCG@5 0,8974 |
| Pertimbangan hukum | Fixed 300/60 | Recall@5 0,8000; MRR@5 0,5279 |
| Recall dakwaan | SAC v1 150/30 | Recall@5 0,7500 |
| Ranking dakwaan | Fixed 500/100 | MRR@5 0,4729; nDCG@5 0,5038 |

Tidak ada dasar untuk menulis “SAC selalu lebih baik daripada fixed”. Klaim yang
didukung adalah: dampak struktur bersifat heterogen menurut section, ukuran
chunk, dan cutoff retrieval.

## 10. Perbandingan SAC v1 dan SAC v2

| Ukuran | Kesimpulan v2 terhadap v1 |
|---:|---|
| 150/30 | Lebih buruk pada seluruh metrik agregat utama dan reasoning |
| 300/60 | Recall@5 naik 0,0125 secara agregat dan reasoning naik 0,0250; MRR/nDCG praktis setara |
| 500/100 | Recall dan MRR agregat lebih rendah; tidak ada keuntungan yang cukup |

SAC v2 tidak layak menggantikan SAC v1 secara umum. Konfigurasi v2 300/60 dapat
dipertahankan sebagai ablation reasoning, tetapi bukan sebagai pemenang utama.
Selection Pure SAC berdasarkan nDCG@5 tetap `sac_w150_o30_s0`.

## 11. Ancaman validitas

1. **Oracle structure:** section berasal dari tag XML Indo-Law, bukan hasil
   detector PDF otomatis. Hasil tidak membuktikan performa SAC pada PDF mentah.
2. **Dataset lama belum direview manusia:** label `approved` dibuat otomatis.
   Dataset v2 sudah lebih konservatif, tetapi masih berstatus draft.
3. **Kueri penahanan tidak seimbang:** hanya satu kueri development pada
   benchmark lama.
4. **Pertanyaan memakai identitas lintas section:** 13 dari 40 draft statute v2
   mempunyai nama query yang tidak muncul pada section evidence. Ini secara
   konstruksi merugikan Pure SAC yang tidak membawa konteks parent.
5. **Development post-hoc:** SAC v2 dikembangkan setelah melihat error; hasilnya
   tidak boleh diperlakukan sebagai konfirmasi.
6. **Holdout lama terkontaminasi:** holdout historis pernah dibuka selama
   perubahan evaluator/desain. Diperlukan holdout baru untuk klaim final.
7. **nDCG masih chunk-based:** overlapping chunk relevan dapat memengaruhi gain,
   walaupun Recall sudah evidence-based.
8. **Latency belum terkontrol:** belum ada warm-up dan repeated online timing
   yang seragam untuk seluruh konfigurasi.

## 12. Narasi hasil yang aman untuk skripsi

Paragraf yang dapat diadaptasi untuk Bab V:

> Evaluasi pada 40 dokumen development menunjukkan bahwa pengaruh
> structure-aware chunking tidak seragam pada seluruh bagian putusan. Pure SAC
> v1 dengan ukuran 150 kata dan overlap 30 kata memperoleh Recall@5 tertinggi
> sebesar 0,6687 dan nDCG@5 tertinggi sebesar 0,4587. Sementara itu, fixed-size
> 500/100 memperoleh Recall@50 tertinggi sebesar 0,9187 dengan jumlah chunk
> paling sedikit. Analisis per bagian menunjukkan bahwa SAC unggul pada
> identitas terdakwa dan amar putusan, sedangkan fixed-size unggul pada
> pertimbangan hukum. Pada pertimbangan hukum, fixed 300/60 mencapai Recall@5
> 0,8000, dibandingkan 0,5500 pada Pure SAC v1 dan 0,5750 pada Pure SAC v2.
> Dengan demikian, hasil tidak menunjukkan keunggulan universal salah satu
> metode, melainkan trade-off antara preservasi struktur, konteks lintas bagian,
> granularitas chunk, dan biaya indeks.

Jangan menulis bahwa perbedaan Recall atau MRR telah signifikan secara umum.
Pada development, CI 95% paired cluster bootstrap untuk Recall@5 dan MRR@5
SAC-vs-fixed masih melintasi nol. Keunggulan yang paling konsisten terlihat pada
nDCG@5 SAC ukuran 150 dan 300, tetapi tetap harus divalidasi pada holdout baru.

## 13. Keputusan metodologis

Untuk skripsi Pure SAC vs fixed-size:

1. laporkan ketiga pasangan ukuran, bukan hanya konfigurasi pemenang;
2. gunakan `fixed_w500_o100` dan `sac_w150_o30_s0` hanya sebagai kandidat terbaik
   per keluarga untuk tahap generasi, bukan sebagai satu-satunya bukti retrieval;
3. pertahankan `sac2_w300_o60_rhet` sebagai ablation reasoning;
4. review 188 draft QA v2 dan bekukan gold baru;
5. gunakan holdout baru yang belum pernah dibuka untuk hasil konfirmatori;
6. laporkan hasil per section karena rerata agregat menyembunyikan perbedaan yang
   sangat besar antara amar, identitas, dakwaan, dan pertimbangan hukum.

Sumber angka:

- `experiments/results/indolaw_200_development_pure_sac_v2_ablation.json`;
- `experiments/results/indolaw_200_development_pure_sac_v2_run_manifest.json`;
- qrels pada `data/evaluation/indolaw_200/`;
- chunks pada `data/chunks/indolaw_200/development/`.
