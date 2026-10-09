# Metodologi B1/B2/M1

## Posisi terhadap Reuter et al. (2025)

Reuter et al. memperkenalkan Summary-Augmented Chunking (SAC): satu ringkasan
dokumen yang sama ditambahkan ke setiap chunk. Mereka tidak mengimplementasikan
ringkasan bagian. Hierarki paragraf-bagian-dokumen hanya disebut sebagai pekerjaan
lanjutan. M1 di sini adalah adaptasi dua tingkat, dokumen dan bagian, bukan algoritme
yang telah divalidasi oleh penulis. Sumber primer: https://aclanthology.org/2025.nllp-1.3/

## Variabel eksperimen

- **B1**: teks chunk asli.
- **B2**: ringkasan dokumen + teks chunk asli.
- **M1**: ringkasan dokumen yang sama dengan B2 + ringkasan bagian + teks chunk asli.

Ketiga metode memakai **19093** chunk canonical yang sama dan ID,
offset, serta bukti sumber yang identik. Alignment terverifikasi:
**True**.

## Chunking

Recursive character splitting versi `rcts-offset-v1` memakai
ukuran 500 karakter, overlap
0, dan separator berurutan
`['\n\n', '\n', '!', '?', '.', ':', ';', ',', ' ', '']`. Penetapan bagian memakai overlap karakter
maksimum; chunk lintas batas disimpan sebagai mixed-section.

## Ringkasan

Model lokal `Irvan14/t5-small-indonesian-summarization` revisi
`65f2d2be6dac02a57444535aca909c5d4606de8a` menghasilkan ringkasan bahasa Indonesia
dengan decoding greedy deterministik. Target 150 karakter dengan toleransi 20.
Bagian yang panjang memakai representasi awal-akhir berbasis token; bagian sangat
pendek disalin secara deterministik. Ringkasan dokumen memakai sampel seimbang
awal-akhir dari enam bagian hukum penting. B2 dan M1 membaca cache ringkasan dokumen
yang sama.

## Embedding dan retrieval

Model `intfloat/multilingual-e5-small` revisi `614241f622f53c4eeff9890bdc4f31cfecc418b3`
memakai prefix `query:`/`passage:`, panjang maksimum 512 token, dan normalisasi L2.
Jika representasi augmented melebihi anggaran, token ringkasan dikurangi terlebih
dahulu; teks asli tidak boleh terpotong. Retrieval memakai exact dense cosine search,
tanpa BM25, reranker, metadata-as-text, atau contextual embedding.

Backend aktual per matriks:

- B1: `sentence_transformers_pytorch` (runtime `2.14.1+cpu`).
- B2: `sentence_transformers_pytorch` (runtime `2.14.1+cpu`).
- M1: `onnxruntime_official_fp32_export` (runtime `1.31.0`).

Smart App Control Windows mengaktifkan blokir DLL PyTorch setelah B1/B2 selesai.
M1 memakai ekspor ONNX FP32 resmi dengan bobot `model.safetensors` yang identik.
Uji kesetaraan ONNX/PyTorch pada 8 input lulus: cosine minimum 1.000000000, selisih absolut maksimum 1.062e-07.

## Evaluasi dan inferensi

Endpoint primer ditetapkan sebelum hasil dibuka: **Recall@5**. Recall memakai unit
bukti sebagai denominator; MRR memakai rank chunk relevan pertama; nDCG memakai
grade 2 untuk chunk yang memuat span lengkap dan grade 1 untuk overlap parsial; DRM
adalah proporsi chunk top-K dari luar dokumen sumber yang benar. Interval keyakinan
95% memakai 2000 bootstrap berpasangan pada
cluster dokumen. Uji tambahan memakai sign-flip cluster dengan
10000 replikasi.
