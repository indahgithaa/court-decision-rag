# Court Decision RAG — Structure-Aware Chunking

Pipeline penelitian untuk membandingkan fixed-size chunking dan
structure-aware chunking (SAC) pada retrieval dan legal QA putusan pengadilan
Indonesia.

## Status penelitian

- Evaluator retrieval aktif memakai schema
  `retrieval-v2-evidence-recall`: Recall@K dihitung pada evidence unit yang sama
  untuk kedua strategi; MRR@K dan nDCG@K dihitung pada ranking chunk.
- Reproduksi Indo-Law 200 selesai secara lokal, tetapi memakai XML dengan gold
  section. Hasilnya adalah oracle-structure robustness experiment.
- Holdout Indo-Law lama pernah dibuka sebelum schema v2 dibekukan. Hasilnya
  eksploratif, bukan konfirmasi blind.
- Runner generasi jawaban dan evaluasi Faithfulness/Answer Relevance/BERTScore
  belum diimplementasikan. Jangan menyatakan penelitian end-to-end selesai.
- `configs/structure_aware_pure_v2.yaml` menyediakan kandidat Pure SAC post-hoc
  dengan boundary retoris source-only. Pada development 300/60, kandidat ini
  hanya memperbaiki Recall reasoning secara kecil dan belum menggantikan Pure
  SAC lama sebagai pemenang agregat.
- `configs/fixed_size_matched.yaml` adalah kontrol 300/60 yang disetarakan
  langsung dengan kandidat Pure SAC v2.
- Generator QA v2 memisahkan tempat lahir dan penahanan, memberi label scope dan
  quality flag, serta selalu menghasilkan status `draft` untuk review manusia.

Audit lengkap ada di `experiments/research_audit.md`; hasil ringkas v2 ada di
`experiments/indolaw_200_reproduction_v2.md`; diagnosis reasoning ada di
`experiments/indolaw_200_reasoning_chunk_ablation.md`; rancangan Pure SAC terbaru
ada di `experiments/pure_sac_v2_design.md`. Perbandingan seluruh konfigurasi,
cutoff, section, interval bootstrap, dan biaya indeks tersedia di
`experiments/fixed_vs_pure_sac_comprehensive_evaluation.md`.

## Setup

Gunakan Python 3.12 untuk menyamai environment reproduksi terakhir.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Reproduksi Indo-Law v2

Jalankan tahap preparasi pada artifact root yang bersih karena script 12–14
meregenerasi file turunannya. Runner retrieval dan evaluator v2 menolak
overwrite hasil/index yang sudah ada. Direktori data, index, dan raw results
diabaikan Git karena ukurannya; simpan salinannya dalam registry terpisah beserta
checksum.

```powershell
# 1. Restore tepat 200 XML dari commit dan hash pada manifest.
.\.venv\Scripts\python.exe scripts\18_restore_indolaw_from_manifest.py

# 2. Bangun corpus, pertanyaan, chunks, dan qrels.
.\.venv\Scripts\python.exe scripts\12_prepare_indolaw_corpus.py
.\.venv\Scripts\python.exe scripts\13_prepare_indolaw_questions.py
.\.venv\Scripts\python.exe scripts\14_build_indolaw_chunk_grid.py

# 3. Development grid, evaluasi, dan freeze desain v2.
.\.venv\Scripts\python.exe scripts\19_run_indolaw_development_grid.py
.\.venv\Scripts\python.exe scripts\15_evaluate_indolaw_design_grid.py

# 3a. Opsional: buat draft QA v2 untuk review manusia (development saja).
.\.venv\Scripts\python.exe scripts\20_prepare_indolaw_questions_v2.py

# 3b. Opsional: ablation Pure SAC v2; tetap development-only.
.\.venv\Scripts\python.exe scripts\14_build_indolaw_chunk_grid.py `
  --splits development --designs sac2_w150_o30_rhet sac2_w300_o60_rhet sac2_w500_o100_rhet
.\.venv\Scripts\python.exe scripts\19_run_indolaw_development_grid.py `
  --configs sac2_w150_o30_rhet sac2_w300_o60_rhet sac2_w500_o100_rhet `
  --manifest-output experiments/results/indolaw_200_development_pure_sac_v2_run_manifest.json
.\.venv\Scripts\python.exe scripts\15_evaluate_indolaw_design_grid.py `
  --configs fixed_w150_o30 sac_w150_o30_s0 sac2_w150_o30_rhet `
            fixed_w300_o60 sac_w300_o60_s0 sac2_w300_o60_rhet `
            fixed_w500_o100 sac_w500_o100_s0 sac2_w500_o100_rhet `
  --output-json experiments/results/indolaw_200_development_pure_sac_v2_ablation.json `
  --output-md experiments/results/indolaw_200_development_pure_sac_v2_ablation.md `
  --no-freeze

# 4. Holdout lama hanya untuk corrected exploratory analysis.
.\.venv\Scripts\python.exe scripts\19_run_indolaw_development_grid.py `
  --split holdout --configs fixed_w500_o100 sac_w150_o30_s0
.\.venv\Scripts\python.exe scripts\16_evaluate_indolaw_holdout.py
```

Model embedding yang dibekukan adalah
`intfloat/multilingual-e5-small`, retrieval corpus-wide top-50, seed 42. Detail
metrik, provenance, hash, dan batas interpretasi tercatat dalam manifest run dan
dokumen eksperimen.

## Struktur penting

- `src/chunking/`: fixed-size dan SAC.
- `src/evaluation/`: qrels, Recall/MRR/nDCG, bootstrap, dan report.
- `src/retrieval/`: dense vector store lokal berbasis NumPy.
- `scripts/`: pipeline data, grid retrieval, evaluasi, dan restore.
- `experiments/`: protokol, design freeze, audit, dan ringkasan hasil.
- `tests/`: unit dan integration tests untuk komponen aktif.

## Aturan klaim

Jangan menyebut hasil Indo-Law sebagai evaluasi PDF/deteksi struktur otomatis,
jangan menyebut holdout lama sebagai blind/final, dan jangan menghidupkan kembali
Hit@K. Sebelum klaim utama, diperlukan validasi manusia atas QA/evidence,
eksperimen PDF atau validasi eksternal yang belum dibuka, serta evaluasi generasi
yang benar-benar diimplementasikan.
