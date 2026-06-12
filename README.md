# 🏛️ RegSearch – Search Engine Regulasi Indonesia

## Deskripsi
Search Engine tematik berbasis **Hybrid Search (BM25 + Dense Retrieval)** untuk mencari
peraturan perundang-undangan Indonesia secara semantik maupun berbasis kata kunci.

**Dataset:** [Indonesian Regulations Dataset – Kaggle](https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset)

## Metode
- **BM25** (Best Match 25) – lexical/keyword matching berbasis probabilistik
- **Dense Retrieval** – semantic search dengan Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Hybrid Score** = `α × BM25_norm + (1-α) × Dense_norm`

## Dataset
- **Sumber**: [Kaggle – Indonesian Regulations Dataset](https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset)
- **Jenis data**: UU, PP, Perpres, Permen, Perda, Kepmen, Surat Edaran
- **Field utama pencarian**: `tentang` (deskripsi narasi panjang per regulasi)
- **Kolom**: `id`, `nomor_peraturan`, `judul`, `jenis_peraturan`, `tentang`, `tahun`
- **Nama file lokal**: `dataset_regulasi_indonesia.csv`

> ⚠️ File dataset **tidak disertakan** dalam ZIP. Unduh dari Kaggle (link di atas)
> atau jalankan `generate_dataset.py` untuk menggunakan data fallback.

## Instalasi

### 1. Extract project
```bash
unzip 220411100183-tugas-UAS.zip
cd search-engine-hukum
```

### 2. Buat virtual environment (disarankan)
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Siapkan dataset
**Opsi A – Dataset Kaggle (direkomendasikan):**
1. Login ke [kaggle.com](https://www.kaggle.com)
2. Unduh dari: https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset
3. Letakkan file CSV hasil unduhan di folder ini
4. Jalankan script normalisasi:
   ```bash
   python generate_dataset.py
   ```

**Opsi B – Data fallback (40 regulasi dummy):**
```bash
python generate_dataset.py
```
Script otomatis membuat `dataset_regulasi_indonesia.csv` dari data bawaan jika file Kaggle tidak ditemukan.

### 5. Jalankan aplikasi
```bash
streamlit run app.py
```
Buka browser: `http://localhost:8501`

## Struktur File
```
search-engine-hukum/
├── app.py                       # Aplikasi Streamlit utama (UI)
├── search_engine.py             # Core: BM25, Dense, Hybrid Search
├── generate_dataset.py          # Persiapan & normalisasi dataset
├── requirements.txt             # Dependencies Python
├── link_dataset.txt             # Link Kaggle dataset
└── README.md                    # Dokumentasi ini
```

## Cara Penggunaan
1. Ketik query di kolom pencarian (contoh: *perlindungan data pribadi*)
2. Atur bobot **Alpha** (BM25 vs Dense) di sidebar
3. Atur jumlah hasil **Top-K**
4. Pilih filter **Jenis Peraturan** (opsional)
5. Klik **🔍 Cari**
6. Lihat kartu hasil dengan skor similarity (Hybrid %, BM25, Dense)
7. Expand tabel & grafik perbandingan skor untuk analisis lebih lanjut
