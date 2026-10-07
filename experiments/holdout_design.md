# Desain holdout final dan robustness eksternal

> **Status 7 Oktober 2026:** desain Indo-Law yang tercatat memerlukan rerun
> dengan schema `retrieval-v2-evidence-recall`. Holdout 160 dokumen lama sudah
> pernah dibuka dan hanya boleh diperlakukan sebagai exploratory; klaim
> konfirmatori memerlukan holdout baru setelah evaluator dan konfigurasi dibekukan.

## Keputusan corpus 200 dokumen

Evaluasi retrieval membandingkan Recall@K, MRR, dan NDCG@K antara
structure-aware chunking dan fixed-size chunking. Corpus eksternal dibekukan
pada **200 dokumen** dan
dibagi berdasarkan pengadilan menjadi 40 dokumen development dan 160 dokumen
holdout. Development digunakan untuk memilih desain dari awal; holdout baru
dibuka setelah konfigurasi dibekukan. Dengan empat pertanyaan per dokumen,
evaluasi terkunci berisi 640 pertanyaan.

Analisis final menggunakan paired cluster bootstrap pada unit dokumen. Seluruh
pertanyaan dari dokumen yang terpilih harus ikut dalam replikasi yang sama.

Corrected analysis menggunakan desain development-frozen `fixed_w500_o100` dan
`sac_w150_o30_s0`. Ringkasan keputusan dan batas klaim tercatat di
`experiments/indolaw_200_research_decision.md`.

## Primary PDF holdout

Klaim utama memerlukan putusan tingkat pertama berformat PDF, klasifikasi
Pid.Sus narkotika, yang melewati pipeline ekstraksi, pembersihan, detector, dan
chunking yang sama dengan pilot. Dokumen pilot dikeluarkan berdasarkan hash;
minimal 70% dokumen berasal dari pengadilan yang tidak ada pada development dan
maksimal dua dokumen diambil per pengadilan. Konfigurasi model, chunking,
reranking dense-only, dan cutoff tidak boleh dituning setelah label holdout
dibuka.

## External normalized-text robustness holdout

Indo-Law dapat dipakai sebagai pengujian robustness eksternal karena berasal
dari Direktori Putusan MA dan menyediakan putusan pidana khusus narkotika dari
banyak pengadilan. Namun XML-nya sudah berupa teks bersih dengan anotasi section,
sehingga distribusi preprocessing berbeda dari PDF pilot. Karena itu hasilnya
harus dilaporkan terpisah dan tidak menggantikan primary PDF holdout.

Sampler `scripts/11_acquire_indolaw_holdout.py` membekukan commit sumber,
mengacak daftar path dengan seed 42, mengecualikan tujuh pengadilan development,
membatasi empat dokumen per pengadilan, memverifikasi lima section wajib, membagi
development/holdout tanpa pengadilan yang tumpang tindih, serta
menyimpan hash dan URL provenance. File XML berada di direktori data yang
diabaikan Git; repository hanya menyimpan manifest metadata untuk menghormati
ketentuan redistribusi dataset.
