# Laporan Penelitian: Hierarchical Summary-Augmented Chunking

## 1. Tujuan penelitian

Penelitian ini menguji apakah penambahan ringkasan tingkat bagian meningkatkan
retrieval bukti pada putusan pengadilan Indonesia dibanding Recursive Chunking (B1)
dan Summary-Augmented Chunking tingkat dokumen (B2). Kontribusi M1 adalah implementasi
dan evaluasi adaptasi dua tingkat, bukan klaim algoritme baru.

## 2. Rujukan ilmiah

Rujukan utama adalah Reuter et al. (2025), *Towards Reliable Retrieval in RAG
Systems for Large Legal Datasets* (https://aclanthology.org/2025.nllp-1.3/). Studi asal memakai ringkasan dokumen
saja, chunk 500 karakter tanpa overlap, dan mengusulkan hierarki ringkasan sebagai
pekerjaan mendatang. Penelitian ini membatasi hierarki pada dokumen dan bagian.

## 3. Karakteristik dataset

Corpus aktif memuat **200 dokumen XML Indo-Law**, bukan PDF, dari
putusan pidana khusus narkotika/psikotropika. Tersedia 800
pertanyaan hasil aturan deterministik. Setelah pemeriksaan otomatis,
**725** dipakai untuk analisis dan
75 ditahan. Tidak ada pertanyaan yang telah
divalidasi manusia; holdout historis juga telah dibuka. Karena itu hasil berikut
adalah **development exploration**, bukan hasil holdout konfirmatori.

## 4. Metodologi

- B1 meng-embed teks asli.
- B2 menambahkan satu ringkasan dokumen yang dipakai ulang untuk seluruh chunk dokumen.
- M1 memakai ringkasan dokumen B2 yang sama dan menambahkan ringkasan bagian terkait.

Sebanyak **19093** chunk canonical dibuat satu kali. Semua
metode mempunyai ID dan batas chunk identik. Penetapan bagian menggunakan overlap
karakter maksimum terhadap tag bagian XML. Embedding menggunakan
`intfloat/multilingual-e5-small` dan cosine similarity.

B1/B2 dihitung dengan PyTorch. Setelah Windows Smart App Control memblokir DLL
PyTorch yang tidak ditandatangani, M1 dilanjutkan memakai ekspor ONNX FP32 resmi dari
bobot model yang sama. Uji pada 8 input menghasilkan cosine minimum 1.000000000 dan selisih absolut maksimum 1.062e-07; status **passed**. Dengan demikian fallback dicatat dan diuji,
bukan diasumsikan ekuivalen.

## 5. Desain eksperimen dan metrik

Endpoint primer adalah Recall@5. Metrik pelengkap ialah MRR@K, nDCG@K, dan DRM@K
untuk K=1,3,5,10; K=20 dan 50 bersifat diagnostik. DRM lebih rendah lebih baik.
Perbandingan berpasangan memakai bootstrap cluster dokumen dan uji sign-flip.

## 6. Hasil retrieval

| Metode | Recall@5 | MRR@5 | nDCG@5 | DRM@5 ↓ |
|---|---:|---:|---:|---:|
| B1 | 0.4979 | 0.3140 | 0.3413 | 0.0913 |
| B2 | 0.5048 | 0.3318 | 0.3518 | 0.0588 |
| M1 | 0.4828 | 0.3900 | 0.3961 | 0.0450 |

## 7. Analisis statistik

| Perbandingan | Metrik | Selisih | 95% CI | p sign-flip |
|---|---|---:|---:|---:|
| B1→B2 | recall@5 | +0.0069 | [-0.0150, +0.0300] | 0.4644 |
| B1→B2 | mrr@5 | +0.0178 | [+0.0039, +0.0324] | 0.0144 |
| B1→B2 | ndcg@5 | +0.0105 | [-0.0021, +0.0239] | 0.1250 |
| B1→B2 | drm@5 | -0.0326 | [-0.0448, -0.0207] | 0.0001 |
| B2→M1 | recall@5 | -0.0221 | [-0.0483, +0.0041] | 0.0761 |
| B2→M1 | mrr@5 | +0.0582 | [+0.0435, +0.0733] | 0.0001 |
| B2→M1 | ndcg@5 | +0.0443 | [+0.0294, +0.0588] | 0.0001 |
| B2→M1 | drm@5 | -0.0138 | [-0.0239, -0.0036] | 0.0061 |
| B1→M1 | recall@5 | -0.0152 | [-0.0415, +0.0098] | 0.3067 |
| B1→M1 | mrr@5 | +0.0760 | [+0.0573, +0.0949] | 0.0001 |
| B1→M1 | ndcg@5 | +0.0548 | [+0.0382, +0.0728] | 0.0001 |
| B1→M1 | drm@5 | -0.0463 | [-0.0604, -0.0318] | 0.0001 |

Kesimpulan endpoint primer B2→M1: selisih M1-B2 -0.0221 dengan CI 95% [-0.0483, +0.0041] melintasi nol; tidak ada bukti peningkatan yang konklusif.

B2−B1 pada Recall@5 adalah +0.0069 dengan CI 95%
[-0.0150, +0.0300] dan p=
0.4644. M1−B2 pada MRR@5 adalah
+0.0582, pada nDCG@5
+0.0443, dan pada DRM@5
-0.0138 (lebih rendah lebih baik). Nilai p masing-masing
0.0001,
0.0001, dan
0.0061.

Nilai p bersifat pelengkap dan tidak mengubah status eksploratori dataset. Interval
yang melintasi nol tidak mendukung klaim peningkatan yang meyakinkan.

## 8. Hasil per bagian

| Bagian | N | B1 Recall@5 | B2 Recall@5 | M1 Recall@5 |
|---|---:|---:|---:|---:|
| `amar_putusan` | 199 | 0.6281 | 0.7186 | 0.8794 |
| `identitas_terdakwa` | 122 | 0.9754 | 1.0000 | 0.9836 |
| `pertimbangan_hukum` | 199 | 0.1055 | 0.1055 | 0.1106 |
| `riwayat_dakwaan` | 199 | 0.4774 | 0.3970 | 0.1608 |
| `riwayat_penahanan` | 6 | 0.1667 | 0.1667 | 0.1667 |

Bagian yang tidak mempunyai pertanyaan valid tidak dinilai; tidak ada angka yang
diimputasi. Distribusi ini terutama menguji identitas, dakwaan, pertimbangan hukum,
amar, dan sedikit riwayat penahanan.

Efek M1 tidak seragam: terhadap B2, Recall@5 amar putusan naik dari
0.7186 menjadi
0.8794, sedangkan riwayat
dakwaan turun dari 0.3970
menjadi 0.1608.

## 9. Diagnostik ringkasan dan kegagalan

Pipeline menghasilkan 200 ringkasan
dokumen dan 2154 ringkasan bagian.
Sebanyak 11 keluaran menerima sedikitnya
satu flag otomatis; sampel pemeriksaan manusia telah disiapkan tetapi belum dinilai.
Token-budget memengaruhi B2 pada
0 chunk dan M1 pada
0 chunk. Semua teks
bukti asli dipertahankan.

Rincian kategori salah dokumen, salah bagian, salah passage, dan bukti di luar top-5
tersedia pada `failure_analysis.json`. Kategori dihitung dari log retrieval, bukan
interpretasi yang dibuat tanpa bukti.

| Kategori top-5 | B1 | B2 | M1 |
|---|---:|---:|---:|
| `correct_document_and_section_wrong_passage` | 186 | 190 | 144 |
| `correct_document_and_section_wrong_passage+evidence_beyond_top_k` | 22 | 39 | 62 |
| `correct_document_wrong_section` | 94 | 83 | 108 |
| `correct_document_wrong_section+evidence_beyond_top_k` | 61 | 47 | 60 |
| `relevant_evidence_in_top_k` | 361 | 366 | 350 |
| `wrong_source_document+evidence_beyond_top_k` | 1 | 0 | 1 |

## 10. Diskusi

Hasil perlu dibaca sebagai efek representasi embedding pada corpus XML yang sangat
terstruktur. B2 dapat mengurangi salah dokumen dengan menambahkan identitas global,
tetapi ringkasan global juga dapat menutupi sinyal lokal. M1 menambah konteks bagian;
manfaatnya bergantung pada ketepatan asosiasi bagian dan kapasitas E5-small untuk
memadatkan tiga komponen ke satu vektor. Arah efek aktual dilaporkan apa adanya dan
tidak dipaksa positif.

Secara agregat, M1 memindahkan lebih banyak bukti relevan ke rank awal (Recall@1,
MRR@5, dan nDCG@5 meningkat), tetapi Recall@5 turun karena kerugian besar pada
pertanyaan dakwaan. Pada K=20/50, Recall M1 kembali melampaui B2. Pola ini mendukung
interpretasi bahwa ringkasan bagian mengubah urutan kandidat secara tajam, bukan
meningkatkan semua tipe bukti secara seragam.

## 11. Keterbatasan dan ancaman validitas

1. Tidak ada PDF dalam corpus aktif, sehingga ekstraksi, OCR, header/footer, dan page
   provenance PDF belum tervalidasi secara empiris untuk eksperimen utama.
2. Tag bagian berasal dari XML Indo-Law dan merupakan kondisi oracle relatif terhadap
   detector otomatis pada PDF.
3. Pertanyaan dan span bukti belum divalidasi manusia. Flag otomatis bukan pengganti
   adjudikasi ahli.
4. Holdout 160 dokumen lama telah dibuka dan dipakai pada eksperimen historis.
5. Model ringkasan lokal berbeda dari GPT-4o-mini pada Reuter et al.; E5-small juga
   berbeda dari GTE-large. Ini adalah substitusi tercatat, bukan reproduksi numerik.
6. Corpus hanya mencakup satu kategori perkara, sehingga generalisasi ke jenis perkara
   lain belum didukung.

## 12. Kesimpulan

Pipeline B1/B2/M1 telah dijalankan dengan chunk dan benchmark yang sama. Temuan saat
ini sah sebagai bukti development eksploratori. Klaim final skripsi tetap menunggu
holdout baru yang dibekukan dan pertanyaan bukti yang divalidasi manusia.

## 13. Reproduksi

Gunakan `configs/hierarchical_sac_2026.yaml` dan jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\27_run_hierarchical_sac.py --config configs\hierarchical_sac_2026.yaml --stage all
```
