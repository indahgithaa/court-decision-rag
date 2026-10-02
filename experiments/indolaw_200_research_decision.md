# Keputusan riset chunking Indo-Law 200

## Kesimpulan

Pada benchmark corpus-wide ini, structure-aware chunking (SAC) meningkatkan
Hit@5 dari 0,5453 menjadi 0,6375 dibanding fixed-size chunking. Selisih
berpasangan sebesar **+0,0922** memiliki interval kepercayaan 95%
**[+0,0469, +0,1375]** berdasarkan 10.000 paired cluster bootstrap pada unit
dokumen. Untuk benchmark ini, desain SAC yang dibekukan adalah pilihan terbaik
untuk retrieval top-5 secara keseluruhan.

Efek tersebut tidak seragam. SAC sangat membantu pertanyaan identitas, dakwaan,
dan amar, tetapi menurunkan Hit@5 pertimbangan hukum dari 0,7562 menjadi 0,1625.
Karena itu, hasil ini mendukung SAC sebagai default keseluruhan, bukan klaim
bahwa SAC terbaik untuk setiap tipe pertanyaan hukum.

## Corpus dan pemisahan data

- 200 putusan pidana khusus narkotika dari Indo-Law, mencakup 71 pengadilan.
- 40 dokumen development dan 160 dokumen holdout, dipisahkan berdasarkan
  pengadilan tanpa pengadilan yang tumpang tindih.
- Maksimum empat dokumen per pengadilan dan empat pertanyaan per dokumen.
- Development berisi 160 pertanyaan; holdout berisi 640 pertanyaan.
- Seluruh file sumber yang dipakai cocok dengan SHA-256 pada manifest.

Corpus memakai teks XML ternormalisasi dan anotasi section dari sumber. Studi
ini dengan demikian mengukur **oracle-structure robustness**, bukan ketahanan
terhadap ekstraksi PDF atau kesalahan deteksi section otomatis.

## Pemilihan desain pada development

Aturan yang dibekukan adalah memaksimalkan Hit@5, kemudian MRR@5, nDCG@5, dan
terakhir memilih jumlah chunk yang lebih kecil. Holdout tidak digunakan untuk
memilih konfigurasi.

| Konfigurasi | Chunk | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| `fixed_w150_o30` | 2.341 | 0,6562 | 0,3622 | 0,3492 |
| `fixed_w300_o60` | 1.173 | 0,6562 | 0,3914 | 0,3779 |
| `fixed_w500_o100` | 710 | 0,6500 | 0,4315 | 0,4168 |
| `sac_w150_o30_s0` | 2.457 | 0,6687 | 0,4146 | 0,4587 |
| `sac_w300_o60_s0` | 1.346 | 0,6500 | 0,4322 | 0,4554 |
| `sac_w500_o100_s0` | 923 | 0,6125 | 0,4255 | 0,4441 |

Desain terpilih adalah `fixed_w300_o60` dan `sac_w150_o30_s0`. Kondisi overlap
dua kalimat tidak menjadi kondisi efektif karena teks XML ternormalisasi tidak
memiliki tanda baca kalimat yang dibutuhkan splitter; keluaran s0 dan s2 identik.

## Hasil holdout terkunci

| Metrik | Fixed-size | SAC | SAC - fixed | 95% CI |
|---|---:|---:|---:|---:|
| Hit@1 | 0,1984 | 0,2687 | +0,0703 | [+0,0359, +0,1047] |
| Hit@3 | 0,4266 | 0,4734 | +0,0469 | [+0,0031, +0,0906] |
| **Hit@5** | **0,5453** | **0,6375** | **+0,0922** | **[+0,0469, +0,1375]** |
| MRR@5 | 0,3245 | 0,3890 | +0,0644 | [+0,0321, +0,0959] |
| nDCG@5 | 0,3066 | 0,4380 | +0,1314 | [+0,1014, +0,1608] |
| Hit@10 | 0,6531 | 0,7406 | +0,0875 | [+0,0484, +0,1281] |
| Hit@50 | 0,8109 | 0,8000 | -0,0109 | [-0,0484, +0,0281] |

Pada Hit@5, kedua strategi berhasil untuk 216 pertanyaan, hanya SAC berhasil
untuk 192, hanya fixed-size berhasil untuk 133, dan keduanya gagal untuk 99.
Hit@50 yang setara menunjukkan keunggulan SAC terutama berasal dari penempatan
bukti lebih awal, bukan peningkatan coverage kandidat pada depth 50.

### Hit@5 per bagian

| Bagian | N | Fixed-size | SAC |
|---|---:|---:|---:|
| Amar putusan | 160 | 0,5062 | 0,7812 |
| Identitas terdakwa | 155 | 0,5032 | 0,9613 |
| Pertimbangan hukum | 160 | 0,7562 | 0,1625 |
| Riwayat dakwaan | 160 | 0,4250 | 0,6750 |
| Riwayat penahanan | 5 | 0,2000 | 0,0000 |

## Batas klaim dan langkah berikutnya

Pertanyaan dan span bukti dibuat dengan aturan deterministik dari XML dan belum
menjadi gold set hasil anotasi manusia. Untuk pertanyaan ketentuan pidana, semua
kemunculan pasal yang ekuivalen di dalam section pertimbangan diberi relevansi
agar strategi tidak dihukum karena mengambil kemunculan yang berbeda.

Klaim yang didukung adalah: SAC oracle-structure lebih baik untuk Hit@5 agregat
pada corpus Indo-Law ternormalisasi ini. Klaim belum mencakup PDF mentah,
deteksi section otomatis, kualitas jawaban generator, atau domain hukum di luar
putusan narkotika. Eksperimen lanjutan sebaiknya menguji routing berbasis tipe
query atau index hibrida—SAC untuk bagian terstruktur dan fixed-size untuk
pertimbangan hukum—dengan pemilihan hanya pada development dan sebuah holdout
baru agar hasil 160 dokumen ini tidak dipakai untuk tuning ulang.
