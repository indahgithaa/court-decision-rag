# Audit dataset Hierarchical SAC 2026

## Status sumber

- Format aktif: **Indo-Law normalized XML**.
- Dokumen XML: **200**; PDF: **0**.
- Dokumen terproses: **200**.
- Error integritas: **0**; XML rusak: **0**; dokumen kosong: **0**.
- Duplikat berbasis SHA-256: **0 kelompok**.
- Penilaian PDF/OCR: not executable: repository contains zero PDFs not applicable: active corpus is normalized XML, not OCR/PDF

Corpus ini tidak mendukung klaim bahwa pipeline PDF telah diuji pada data utama.

## Struktur dokumen

| Label bagian | Jumlah |
|---|---:|
| `amar_putusan` | 200 |
| `fakta` | 200 |
| `fakta_hukum` | 159 |
| `identitas_terdakwa` | 200 |
| `kepala_putusan` | 200 |
| `penutup` | 199 |
| `pertimbangan_hukum` | 200 |
| `riwayat_dakwaan` | 200 |
| `riwayat_penahanan` | 200 |
| `riwayat_perkara` | 199 |
| `riwayat_tuntutan` | 197 |

Semua posisi bagian diverifikasi terhadap teks dokumen. Jumlah error rekonstruksi:
**0**.

## Benchmark QA

- Pertanyaan tersedia: **800**.
- Lolos pemeriksaan otomatis untuk analisis development: **725**.
- Ditahan untuk pemeriksaan manusia: **75**.
- Anotasi tervalidasi manusia: **0**.
- Layak konfirmatori: **0**.

| Flag otomatis | Jumlah |
|---|---:|
| `excessive_question_length` | 4 |
| `field_label_as_answer` | 26 |
| `implausibly_short_answer` | 44 |
| `malformed_identity_question` | 13 |

Label berasal dari aturan deterministik atas XML, bukan anotasi manusia. Split holdout
lama telah dibuka dan dipakai berulang, sehingga seluruh evaluasi baru wajib disebut
eksperimen development eksploratori.
