# Protokol Evaluasi Jawaban Indo-Law 200

## Tujuan

Membandingkan dampak *fixed-size chunking* dan *structure-aware chunking* (SAC)
terhadap kualitas jawaban, dengan komponen selain strategi chunking dibuat sama.

## Desain yang dibekukan

- Fixed-size: `fixed_w500_o100`
- SAC: `sac_w150_o30_s0`
- Retriever: `intfloat/multilingual-e5-small`, dense-only
- Prompt: `legal_qa_id_v1`
- Kandidat anggaran konteks development: 600, 900, 1.200, 1.500, 1.800, dan 2.400 kata
- Urutan konteks: berdasarkan peringkat retrieval; chunk terakhir dipotong bila perlu
- Jawaban acuan tidak boleh dimasukkan ke prompt generasi

Anggaran kata yang sama mencegah fixed-size memperoleh konteks lebih banyak hanya
karena ukuran chunk terpilihnya lebih besar. Label bagian SAC tidak dimasukkan ke prompt
agar struktur metadata tidak menjadi perlakuan tambahan. Setelah model dipilih, jumlah
token prompt juga harus diaudit karena jumlah kata yang sama tidak selalu menghasilkan
jumlah token yang identik. Anggaran final dipilih pada development berdasarkan kualitas
jawaban dan efisiensi, kemudian dibekukan sebelum evaluasi holdout.

## Tahapan eksperimen

1. Gunakan development (40 dokumen) untuk memvalidasi prompt, kegagalan parsing,
   panjang konteks, dan konfigurasi model.
2. Bekukan model, versi model, parameter decoding, prompt, dan evaluator.
3. Jalankan kedua strategi dengan model serta konfigurasi yang sama. Gunakan suhu 0
   jika penyedia model mendukungnya. Acak urutan pertanyaan dengan seed 42 dan
   selang-seling strategi yang dijalankan lebih dahulu di dalam setiap pasangan.
4. Buka holdout (160 dokumen) hanya setelah seluruh konfigurasi dibekukan.
5. Jangan memilih ulang desain chunking atau prompt berdasarkan hasil holdout.

## Metrik

- Faithfulness (Ragas 0.4.3)
- Answer Relevance (Ragas 0.4.3)
- BERTScore (BERTScore 0.3.13; jawaban acuan sebagai pembanding)
- Latensi generasi per pertanyaan dalam milidetik

Laporkan rerata tiap strategi, selisih SAC dikurangi fixed-size, serta 95% confidence
interval dari paired cluster bootstrap pada tingkat dokumen. Retrieval latency dan
generation latency disimpan terpisah; latensi end-to-end adalah jumlah keduanya ditambah
waktu perakitan prompt. Lakukan warm-up dan jalankan secara sekuensial atau dengan
concurrency identik agar perbandingan latensi dapat ditafsirkan.

## Keputusan yang belum dibekukan

Model generatif dan model penilai Ragas belum ditentukan. Keduanya harus dicatat dengan
nama serta versi yang dapat direproduksi, dan tidak boleh berbeda antarstrategi.
