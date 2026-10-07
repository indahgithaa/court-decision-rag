# Keputusan riset chunking Indo-Law 200

> **Status 7 Oktober 2026: INVALID / NEEDS RERUN.** Angka di bawah berasal dari
> Recall dengan denominator chunk relevan yang dapat berbeda antardesain. Setelah
> evaluator dipindahkan ke unit evidence bersama, development selection dan
> holdout harus dijalankan ulang. Pertahankan bagian ini hanya sebagai catatan
> corrected exploratory analysis lama, bukan hasil final.

## Kesimpulan

Pada corrected analysis corpus-wide, structure-aware chunking (SAC) memberikan
Recall@5 dan NDCG@5 yang lebih tinggi daripada fixed-size chunking. Recall@5
meningkat dari 0,4322 menjadi 0,6080, sedangkan NDCG@5 meningkat dari 0,3476
menjadi 0,4380. Interval kepercayaan 95% untuk kedua selisih berada di atas nol.

MRR@5 meningkat dari 0,3574 menjadi 0,3890, tetapi interval selisihnya sedikit
melintasi nol. Karena itu bukti peningkatan posisi hasil relevan pertama pada
cutoff 5 belum konklusif. Efek juga berbeda antarbagian: SAC sangat membantu
amar dan identitas, sedangkan fixed-size tetap lebih kuat untuk pertimbangan
hukum.

## Corpus dan pemisahan data

- 200 putusan pidana khusus narkotika dari Indo-Law, mencakup 71 pengadilan.
- 40 dokumen development dan 160 dokumen holdout, dipisahkan berdasarkan
  pengadilan tanpa pengadilan yang tumpang tindih.
- Maksimum empat dokumen per pengadilan dan empat pertanyaan per dokumen.
- Development berisi 160 pertanyaan; holdout berisi 640 pertanyaan.
- Seluruh file sumber yang dipakai cocok dengan SHA-256 pada manifest.

Corpus memakai teks XML ternormalisasi dan anotasi section dari sumber. Studi
ini mengukur **oracle-structure robustness**, bukan ketahanan terhadap ekstraksi
PDF atau kesalahan deteksi section otomatis.

## Pemilihan desain pada development

Evaluator mengikuti metrik pada draft skripsi: Recall@K, MRR, dan NDCG@K.
Aturan operasional untuk memilih satu desain adalah NDCG@5, kemudian MRR@5,
Recall@5, dan terakhir jumlah chunk yang lebih kecil.

| Konfigurasi | Chunk | Recall@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| `fixed_w150_o30` | 2.341 | 0,5128 | 0,3622 | 0,3492 |
| `fixed_w300_o60` | 1.173 | 0,5192 | 0,3914 | 0,3779 |
| `fixed_w500_o100` | 710 | 0,5205 | 0,4315 | 0,4168 |
| `sac_w150_o30_s0` | 2.457 | 0,6285 | 0,4146 | 0,4587 |
| `sac_w300_o60_s0` | 1.346 | 0,5707 | 0,4322 | 0,4554 |
| `sac_w500_o100_s0` | 923 | 0,5445 | 0,4255 | 0,4441 |

Desain terpilih adalah `fixed_w500_o100` dan `sac_w150_o30_s0`. Kondisi overlap
dua kalimat tidak menjadi kondisi efektif karena teks XML ternormalisasi tidak
memiliki tanda baca kalimat yang dibutuhkan splitter; keluaran s0 dan s2 identik.

## Hasil corrected holdout analysis

| Metrik | Fixed-size | SAC | SAC - fixed | 95% CI |
|---|---:|---:|---:|---:|
| Recall@1 | 0,2154 | 0,2654 | +0,0501 | [+0,0191, +0,0803] |
| MRR@1 | 0,2437 | 0,2687 | +0,0250 | [-0,0109, +0,0594] |
| NDCG@1 | 0,2437 | 0,2687 | +0,0250 | [-0,0109, +0,0594] |
| Recall@5 | 0,4322 | 0,6080 | +0,1758 | [+0,1319, +0,2174] |
| MRR@5 | 0,3574 | 0,3890 | +0,0316 | [-0,0032, +0,0647] |
| NDCG@5 | 0,3476 | 0,4380 | +0,0904 | [+0,0570, +0,1223] |
| Recall@10 | 0,5074 | 0,6929 | +0,1855 | [+0,1452, +0,2242] |
| MRR@10 | 0,3686 | 0,4038 | +0,0352 | [+0,0015, +0,0673] |
| NDCG@10 | 0,3743 | 0,4678 | +0,0935 | [+0,0623, +0,1231] |

### Metrik @5 per bagian

| Bagian | N | Fixed Recall | SAC Recall | Fixed MRR | SAC MRR | Fixed NDCG | SAC NDCG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amar putusan | 160 | 0,2750 | 0,7812 | 0,1652 | 0,3214 | 0,1880 | 0,4348 |
| Identitas terdakwa | 155 | 0,7355 | 0,9613 | 0,5676 | 0,9454 | 0,6096 | 0,9493 |
| Pertimbangan hukum | 160 | 0,1976 | 0,0445 | 0,3139 | 0,0602 | 0,1853 | 0,0367 |
| Riwayat dakwaan | 160 | 0,5250 | 0,6750 | 0,3937 | 0,2584 | 0,4169 | 0,3608 |
| Riwayat penahanan | 5 | 0,6000 | 0,0000 | 0,2167 | 0,0000 | 0,3123 | 0,0000 |

## Batas klaim dan status penelitian

Pertanyaan dan span bukti dibuat dengan aturan deterministik dari XML dan belum
menjadi gold set hasil anotasi manusia. Untuk pertanyaan ketentuan pidana, semua
kemunculan pasal yang ekuivalen di dalam section pertimbangan diberi relevansi.

Holdout telah dibuka sebelum koreksi metrik dilakukan. Walaupun desain baru
dipilih ulang secara mekanis hanya dari development, hasil ini harus disebut
**corrected analysis of a previously opened holdout**, bukan validasi
konfirmatori baru yang sepenuhnya buta.

Klaim sementara yang didukung adalah SAC oracle-structure meningkatkan coverage
dan kualitas ranking agregat pada corpus Indo-Law ternormalisasi. Penelitian
skripsi belum selesai karena evaluasi kualitas jawaban (Faithfulness, Answer
Relevance, dan BERTScore), evaluasi latency yang direncanakan, serta validasi
pada pipeline PDF/deteksi section otomatis belum lengkap.
