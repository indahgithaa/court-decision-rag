# Perancangan Hybrid Structure-Aware dan Summary-Augmented Chunking

## Status penelitian

Penelitian aktif memakai **satu corpus utuh**, bukan pembagian
development/holdout. Fixed-size, structure-aware, dan hybrid dievaluasi pada
collection yang sama, yaitu 200 dokumen dan 800 pertanyaan. Karena bentuk hybrid
dikembangkan melalui eksperimen terdahulu pada dokumen yang sama, hasil harus
disebut **evaluasi komparatif eksploratif**, bukan estimasi performa pada data
baru yang belum pernah diamati.

## Landasan metode

Metode hybrid menyatukan tiga ide yang sudah memiliki dukungan publikasi:

1. *Structure-Aware Chunking* memisahkan putusan berdasarkan strata retoris
   sebelum pemrosesan lanjutan (Sonowal dan Sadhu, 2025,
   DOI `10.18653/v1/2025.justnlp-main.19`).
2. *Summary-Augmented Chunking* membuat satu *document fingerprint* ringkas dan
   menambahkannya ke setiap chunk sebelum embedding untuk mengurangi
   *Document-Level Retrieval Mismatch* (Reuter dkk., 2025,
   DOI `10.18653/v1/2025.nllp-1.3`).
3. Penandaan peran retoris secara eksplisit pada input telah digunakan untuk
   memelihara informasi struktur putusan (Arjun T D dan Madasamy, 2025,
   DOI `10.18653/v1/2025.justnlp-main.12`).

Kombinasi spesifik ketiganya serta konteks hierarkis dokumen--bagian adalah
kontribusi/adaptasi penelitian ini, bukan klaim replikasi persis salah satu
paper.

## Arsitektur final

Nama desain final: `hybrid_hier_w150_o30_s0`.

1. Putusan dibagi mengikuti batas bagian pada XML: identitas, penahanan,
   dakwaan, fakta, pertimbangan hukum, amar, dan bagian lain yang tersedia.
2. Isi setiap bagian dibagi menjadi window maksimum 150 kata dengan overlap
   30 kata. Chunk tidak menyeberangi batas bagian.
3. Dibuat konteks dokumen yang stabil: nomor perkara, nama terdakwa, dan topik
   perkara. Konteks ini sama untuk seluruh chunk dalam satu putusan.
4. Dibuat konteks bagian yang berbeda menurut peran retoris. Contohnya, pasal
   dominan hanya ditambahkan pada konteks `pertimbangan_hukum`, sedangkan jenis
   pidana hanya ditambahkan pada konteks `amar_putusan`.
5. Teks embedding disusun sebagai `konteks dokumen + konteks bagian + teks
   chunk`. Teks bukti, offset karakter, dan qrels tetap menunjuk teks sumber
   verbatim.
6. Embedding menggunakan `intfloat/multilingual-e5-small` dengan prefix E5,
   normalisasi vektor, dan dense retrieval pada seluruh korpus.

Fingerprint pada eksperimen ini dibuat dengan ekstraksi deterministik yang
bersumber dari teks putusan. Reuter dkk. menggunakan ringkasan generatif
GPT-4o-mini sekitar 150 karakter; karena itu implementasi ini adalah adaptasi
hemat sumber daya, bukan replikasi identik.

## Desain evaluasi corpus tunggal

Ketiga metode memakai konfigurasi ukuran yang sama, yaitu maksimum 150 kata dan
overlap 30 kata. Konfigurasi ini dipakai karena sesuai batas input encoder E5;
audit seluruh 12.206 input hybrid menghasilkan maksimum 379 token, P95 279
token, dan tidak ada input yang melewati 512 token. Semua 800 kueri melakukan
retrieval terhadap indeks yang berisi seluruh 200 dokumen.

Tidak terdapat data latih, data validasi, atau data uji terpisah. Metode tidak
melakukan pelatihan model pada Indo-Law. Interval kepercayaan dihitung memakai
paired cluster bootstrap 10.000 iterasi dengan dokumen sebagai unit sampling.

## Hasil evaluasi keseluruhan

| Desain | Chunk | Recall@5 | MRR@5 | nDCG@5 | DRM@5 (turun lebih baik) |
|---|---:|---:|---:|---:|---:|
| Fixed 150/30 | 11.605 | 0,6350 | 0,3638 | 0,3489 | 0,0620 |
| Structure-aware 150/30 | 12.206 | 0,6425 | 0,3935 | 0,4414 | 0,1028 |
| **Hybrid hierarkis 150/30** | **12.206** | **0,6613** | **0,4510** | **0,4522** | **0,0262** |

Dibanding fixed-size, hybrid memperoleh:

- selisih Recall@5 `+0,0262`, CI 95% `[-0,0138, +0,0650]`;
- selisih MRR@5 `+0,0871`, CI 95% `[+0,0580, +0,1157]`;
- selisih nDCG@5 `+0,1033`, CI 95% `[+0,0777, +0,1284]`; dan
- selisih DRM@5 `-0,0357`, CI 95% `[-0,0495, -0,0237]`.

Dibanding structure-aware tanpa augmentasi, hybrid memperoleh selisih Recall@5
`+0,0187` (CI masih melintasi nol), MRR@5 `+0,0574` (CI tidak melintasi nol),
nDCG@5 `+0,0108` (CI masih melintasi nol), dan DRM@5 `-0,0765` (CI tidak
melintasi nol).

Dengan demikian, bukti terkuat berada pada perbaikan MRR, nDCG terhadap fixed,
dan penurunan mismatch dokumen. Kenaikan Recall@5 bersifat deskriptif karena
interval kepercayaannya masih melintasi nol.

## Hasil Recall@5 per bagian

| Bagian | N | Fixed | Structure-aware | Hybrid | Metode tertinggi |
|---|---:|---:|---:|---:|---|
| Amar putusan | 200 | 0,6400 | **0,7900** | 0,5850 | Structure-aware |
| Identitas terdakwa | 194 | 0,6495 | 0,9485 | **0,9794** | Hybrid |
| Pertimbangan hukum | 200 | **0,7400** | 0,1750 | 0,6450 | Fixed |
| Riwayat dakwaan | 200 | 0,5150 | **0,6850** | 0,4550 | Structure-aware |
| Riwayat penahanan | 6 | **0,5000** | 0,0000 | 0,3333 | Fixed |

Hybrid tidak mendominasi seluruh bagian. Pada pertimbangan hukum, hybrid jauh
lebih baik daripada structure-aware murni tetapi masih di bawah fixed-size.
Kesimpulan yang sah adalah konteks hierarkis meningkatkan kualitas ranking
agregat dan keterlacakan dokumen, bukan bahwa hybrid selalu unggul pada setiap
jenis pertanyaan.

## Batas validitas

- Batas bagian berasal dari anotasi/struktur XML yang tersedia, bukan hasil
  deteksi otomatis langsung dari PDF. Performa end-to-end pada PDF mentah belum
  diukur.
- Beberapa XML/OCR menyusun label dan nilai identitas secara tidak teratur.
  Ekstraktor deterministik mengurangi, tetapi tidak menghapus, kesalahan nama
  pada fingerprint.
- Pertanyaan benchmark lama juga memiliki sejumlah nama terdakwa yang rusak
  akibat aturan ekstraksi awal. Hasil perlu dibaca bersama audit dataset.
- Pertanyaan riwayat penahanan hanya berjumlah enam, sehingga hasil bagian itu
  tidak cukup stabil untuk dijadikan kesimpulan utama.
- Hanya satu encoder multilingual kecil yang diuji. Generalisasi ke embedding
  legal-domain atau model lain belum dapat diklaim.
- Corpus yang sama ikut memengaruhi evolusi desain. Karena itu, hasil tidak
  boleh disebut performa test-set independen atau bukti generalisasi. Validasi
  eksternal dapat ditambahkan sebagai penelitian lanjutan tanpa membagi corpus
  Indo-Law 200 yang digunakan dalam analisis komparatif ini.

## Artefak utama

- Hasil lengkap: `experiments/results/indolaw_200_corpus_hybrid_final.json`
- Laporan: `experiments/results/indolaw_200_corpus_hybrid_final.md`
- Manifest indexing:
  `experiments/results/indolaw_200_corpus_hybrid_run_manifest.json`
- Ringkasan versionable: `experiments/hybrid_chunking_final_results.json`

