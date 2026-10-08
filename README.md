# Court Decision RAG — Hybrid Structure- and Summary-Augmented Chunking

Pipeline penelitian retrieval dan legal QA putusan pengadilan Indonesia.
Eksperimen terbaru menggabungkan structure-aware chunking, document
fingerprint, dan konteks bagian retoris secara hierarkis.

## Status penelitian

- Evaluator retrieval aktif memakai schema
  `retrieval-v2-evidence-recall`: Recall@K dihitung pada evidence unit yang sama
  untuk kedua strategi; MRR@K dan nDCG@K dihitung pada ranking chunk.
- Reproduksi Indo-Law 200 selesai secara lokal, tetapi memakai XML dengan gold
  section. Hasilnya adalah oracle-structure robustness experiment.
- Desain skripsi aktif tidak membagi corpus menjadi development/holdout.
  Fixed-size, structure-aware, dan hybrid dibandingkan pada satu corpus yang
  sama: 200 dokumen dan 800 pertanyaan. Karena desain dikembangkan dengan
  corpus tersebut, hasilnya bersifat komparatif eksploratif.
- Pada corpus tunggal, `hybrid_hier_w150_o30_s0` mencapai Recall@5 0,6613,
  MRR@5 0,4510, nDCG@5 0,4522, dan DRM@5 0,0262. Hybrid memperbaiki ranking
  agregat dan provenance dokumen, tetapi tidak mengungguli baseline pada semua
  bagian retoris.
- Runner generasi jawaban dan evaluasi Faithfulness/Answer Relevance/BERTScore
  belum diimplementasikan. Jangan menyatakan penelitian end-to-end selesai.
- Generator QA v2 memisahkan tempat lahir dan penahanan, memberi label scope dan
  quality flag, serta selalu menghasilkan status `draft` untuk review manusia.

Audit lengkap ada di `experiments/research_audit.md`; hasil ringkas v2 ada di
`experiments/indolaw_200_reproduction_v2.md`; diagnosis reasoning ada di
`experiments/indolaw_200_reasoning_chunk_ablation.md`; rancangan Pure SAC terbaru
ada di `experiments/pure_sac_v2_design.md`. Perbandingan seluruh konfigurasi,
cutoff, section, interval bootstrap, dan biaya indeks tersedia di
`experiments/fixed_vs_pure_sac_comprehensive_evaluation.md`.
Perancangan dan hasil hybrid terbaru ada di
`experiments/hybrid_structure_summary_design.md`, dengan ringkasan numerik di
`experiments/hybrid_chunking_final_results.json`.

## Setup

Gunakan Python 3.12 untuk menyamai environment reproduksi terakhir.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Reproduksi eksperimen corpus tunggal

Pipeline aktif memakai tepat satu collection bernama `corpus`. Script menolak
overwrite artefak retrieval yang sudah ada. Direktori data, index, dan raw
results diabaikan Git karena ukurannya; simpan salinannya beserta checksum.

```powershell
# 1. Restore 200 XML, lalu bangun satu corpus dan satu set pertanyaan.
.\.venv\Scripts\python.exe scripts\18_restore_indolaw_from_manifest.py
.\.venv\Scripts\python.exe scripts\12_prepare_indolaw_corpus.py
.\.venv\Scripts\python.exe scripts\13_prepare_indolaw_questions.py

# 2. Buat konteks dokumen dan konteks bagian yang source-grounded.
.\.venv\Scripts\python.exe scripts\21_generate_indolaw_summaries.py `
  --collections corpus `
  --output-root data/processed/indolaw_200/document_summaries_corpus

# 3. Bangun tiga metode yang dibandingkan pada konfigurasi matched 150/30.
.\.venv\Scripts\python.exe scripts\14_build_indolaw_chunk_grid.py `
  --collections corpus `
  --summaries-root data/processed/indolaw_200/document_summaries_corpus `
  --designs fixed_w150_o30 sac_w150_o30_s0 hybrid_hier_w150_o30_s0

# 4. Bangun indeks dan jalankan semua 800 kueri terhadap 200 dokumen.
.\.venv\Scripts\python.exe scripts\19_run_indolaw_development_grid.py `
  --collection corpus `
  --configs fixed_w150_o30 sac_w150_o30_s0 hybrid_hier_w150_o30_s0 `
  --manifest-output experiments/results/indolaw_200_corpus_hybrid_run_manifest.json

# 5. Hitung metrik agregat, per bagian, dan paired document bootstrap.
.\.venv\Scripts\python.exe scripts\22_evaluate_hybrid_chunking.py `
  --collection corpus `
  --configs fixed_w150_o30 sac_w150_o30_s0 hybrid_hier_w150_o30_s0 `
  --evaluation-stage single_corpus_exploratory `
  --output-json experiments/results/indolaw_200_corpus_hybrid_final.json `
  --output-md experiments/results/indolaw_200_corpus_hybrid_final.md
```

Model embedding yang dibekukan adalah
`intfloat/multilingual-e5-small`, retrieval corpus-wide top-50, seed 42. Detail
metrik, provenance, hash, dan batas interpretasi tercatat dalam manifest run dan
dokumen eksperimen.

## Struktur penting

- `src/chunking/`: fixed-size, structure-aware, dan summary augmentation.
- `src/evaluation/`: qrels, Recall/MRR/nDCG, bootstrap, dan report.
- `src/retrieval/`: dense vector store lokal berbasis NumPy.
- `scripts/`: pipeline data, grid retrieval, evaluasi, dan restore.
- `experiments/`: protokol, design freeze, audit, dan ringkasan hasil.
- `tests/`: unit dan integration tests untuk komponen aktif.

## Aturan klaim

Jangan menyebut hasil Indo-Law sebagai evaluasi PDF/deteksi struktur otomatis,
hasil test-set independen, atau bukti generalisasi ke data baru. Jangan
menghidupkan kembali Hit@K. Hasil aktif adalah evaluasi komparatif eksploratif
pada satu corpus. Klaim generalisasi memerlukan validasi manusia atas QA/evidence
dan evaluasi eksternal terpisah; itu tidak mengharuskan corpus Indo-Law 200 ini
dibagi menjadi development/holdout.
