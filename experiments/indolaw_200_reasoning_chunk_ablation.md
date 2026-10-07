# Ablation SAC untuk `pertimbangan_hukum`

Tanggal: 7 Oktober 2026  
Split: development, 40 dokumen / 160 pertanyaan  
Status: **post-hoc development diagnostic — bukan hasil konfirmatori**

## Diagnosis

SAC lama menghormati batas section, tetapi menyematkan hanya teks lokal chunk.
Pertanyaan ketentuan pidana memuat nama terdakwa, sedangkan banyak chunk
`pertimbangan_hukum` berisi uraian pasal generik tanpa nama. Dense retriever
akhirnya memilih chunk `amar_putusan`, `riwayat_tuntutan`, atau
`riwayat_dakwaan` dari dokumen yang benar karena bagian-bagian itu memuat nama
terdakwa secara eksplisit.

Pada SAC lama, top-1 untuk 40 pertanyaan hukum berasal dari:

- `amar_putusan`: 18;
- `riwayat_tuntutan`: 18;
- `pertimbangan_hukum`: 3;
- `riwayat_dakwaan`: 1.

Section `pertimbangan_hukum` juga memiliki median 1.911,5 kata. Karena XML
ternormalisasi hampir tidak memiliki tanda baca kalimat, sentence-aware splitter
jatuh menjadi word windows. Reset window di batas section dapat menghasilkan
tail chunk pendek; contoh yang diaudit hanya berisi 43 kata dan berada pada rank
40, sementara fixed window dengan konteks lebih panjang berada pada rank 1.

## Kandidat yang diuji

Semua kandidat memakai 150 kata, overlap 30 kata, 2.457 chunk, model
`intfloat/multilingual-e5-small`, dan qrel/evidence unit yang sama kecuali varian
tail yang menambah chunk relevan ekuivalen akibat coverage ujung section.

1. `tail`: anchor ulang final chunk yang panjangnya ≤50% budget agar berakhir
   dengan konteks penuh tanpa menyeberangi batas section.
2. `head`: prepend label section hanya pada `embedding_text`; teks sumber dan
   offset tidak berubah.
3. `ctx`: prepend label section dan identitas terdakwa pada semua chunk.
4. `adaptive`: prepend label section pada semua chunk, tetapi identitas parent
   hanya pada section analitis `pertimbangan_hukum` dan `fakta_hukum`.

## Hasil development

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| SAC lama | 0.6687 | 0.4146 | 0.4587 | 0.7250 | 0.8125 |
| Tail backfill | 0.6562 | 0.4115 | 0.4407 | 0.7250 | 0.8187 |
| Section heading | 0.7000 | 0.5365 | 0.5594 | 0.7438 | 0.8063 |
| Heading + identitas semua chunk | 0.6188 | 0.4501 | 0.4601 | 0.7937 | 0.8938 |
| **Adaptive hierarchical context** | **0.7312** | **0.5504** | **0.5626** | **0.8187** | **0.8938** |

Untuk `pertimbangan_hukum`:

| Konfigurasi | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| SAC lama | 0.2250 | 0.0958 | 0.0530 |
| Tail backfill | 0.2250 | 0.0946 | 0.0603 |
| Section heading | 0.2000 | 0.1021 | 0.0561 |
| Heading + identitas semua chunk | 0.4250 | 0.1812 | 0.1164 |
| **Adaptive hierarchical context** | **0.5250** | **0.2413** | **0.1822** |

Paired cluster bootstrap 10.000 sampel pada unit dokumen untuk adaptive minus
SAC lama:

| Scope/metrik | Delta | 95% CI |
|---|---:|---:|
| Keseluruhan Recall@5 | +0.0625 | [+0.0063, +0.1187] |
| Keseluruhan MRR@5 | +0.1358 | [+0.1004, +0.1702] |
| Keseluruhan nDCG@5 | +0.1039 | [+0.0710, +0.1379] |
| Pertimbangan Recall@5 | +0.3000 | [+0.1250, +0.4750] |
| Pertimbangan MRR@5 | +0.1454 | [+0.0737, +0.2254] |
| Pertimbangan nDCG@5 | +0.1292 | [+0.0700, +0.1903] |

Pada adaptive, top-1 `pertimbangan_hukum` naik dari 3/40 menjadi 12/40 dan
Recall@10 naik dari 0,300 menjadi 0,700. Namun Recall@5 `riwayat_dakwaan` turun
dari 0,750 menjadi 0,600 karena chunk reasoning menjadi lebih kompetitif.

## Keputusan

- Tail backfill tersedia sebagai opsi tetapi **ditolak sebagai default** karena
  tidak meningkatkan Recall@5 reasoning dan menurunkan hasil keseluruhan.
- `section_reasoning_document` adalah kandidat terbaik pada development dan
  tersedia melalui `configs/structure_aware_adaptive.yaml`.
- `text`, `start_position`, dan `end_position` tetap berasal dari sumber asli;
  contextual prefix hanya berada di `embedding_text` dan tidak masuk prompt
  jawaban.
- Frozen design v2 tidak diubah. Kandidat ini ditemukan setelah holdout lama
  pernah dilihat, sehingga tidak boleh diuji untuk klaim konfirmatori pada
  holdout tersebut. Validasi berikutnya harus memakai holdout baru atau dataset
  eksternal yang belum dibuka.
