"""
=============================================================
  HYBRID SEARCH ENGINE - REGULASI INDONESIA
  Metode  : BM25 + Dense Retrieval (Sentence Transformers)
  Dataset : Indonesian Regulations Dataset (Kaggle)
  URL     : https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset
  Author  : 220411100183 - Tugas UAS Temu Kembali Informasi
=============================================================
"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import re
import os
import pickle

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESSING DATASET
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset CSV regulasi Indonesia dari Kaggle.

    Field utama yang digunakan untuk pencarian adalah kolom 'tentang'
    karena mengandung deskripsi/narasi panjang tentang isi peraturan —
    sesuai ketentuan tugas yang mengharuskan field berisi kalimat panjang
    (bukan field 3 kata seperti nama produk/tempat).

    Sumber dataset:
      https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Pastikan field utama pencarian tidak kosong
    df = df.dropna(subset=["tentang"])
    df = df[df["tentang"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"[Dataset] Dimuat: {len(df)} regulasi dari {filepath}")
    return df


def tokenize(text: str) -> list[str]:
    """
    Tokenisasi teks untuk BM25:
      1. Lowercase — agar pencarian case-insensitive
      2. Hapus tanda baca dan karakter non-alfanumerik
      3. Split berdasarkan spasi

    Contoh:
      Input : "Perlindungan Data Pribadi (UU No. 27/2022)"
      Output: ["perlindungan", "data", "pribadi", "uu", "no", "27", "2022"]
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # Hapus tanda baca
    text = re.sub(r"\s+", " ", text)        # Hapus spasi berlebih
    tokens = text.split()
    return tokens


# ─────────────────────────────────────────────
# 2. MEMBANGUN INDEKS BM25
# ─────────────────────────────────────────────

def build_bm25_index(corpus: list[str]) -> BM25Okapi:
    """
    Membangun indeks BM25 (Best Match 25) dari corpus teks regulasi.

    BM25 adalah algoritma ranking dokumen berbasis probabilistik yang
    memperhitungkan:
      - TF  (Term Frequency)    : seberapa sering kata muncul di dokumen
      - IDF (Inverse Doc Freq)  : seberapa langka kata di seluruh corpus
      - Panjang dokumen         : normalisasi terhadap panjang rata-rata

    Parameter BM25Okapi default:
      k1 = 1.5  → saturasi frekuensi kata
      b  = 0.75 → normalisasi panjang dokumen

    Proses:
      1. Tokenisasi seluruh corpus
      2. Buat indeks BM25 dari token-token tersebut
    """
    print("[BM25] Membangun indeks BM25...")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[BM25] Indeks selesai: {len(tokenized_corpus)} dokumen diindeks")
    return bm25


# ─────────────────────────────────────────────
# 3. MEMBANGUN INDEKS DENSE (SENTENCE EMBEDDING)
# ─────────────────────────────────────────────

def build_dense_index(corpus: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Menghasilkan vektor embedding (Dense Index) untuk setiap teks regulasi
    menggunakan model Sentence Transformers multilingual.

    Model yang digunakan:
      'paraphrase-multilingual-MiniLM-L12-v2'
      - Mendukung 50+ bahasa termasuk Bahasa Indonesia
      - Dimensi embedding: 384 (vektor float32)
      - Ringan (~120MB) dan cepat di CPU

    Proses:
      1. Setiap teks regulasi di-encode menjadi vektor 384-dimensi
      2. Vektor disimpan dalam array numpy (n_doc × 384)
      3. Digunakan untuk menghitung cosine similarity saat query

    Output shape: (jumlah_dokumen, 384)
    """
    print(f"[Dense] Membuat embedding untuk {len(corpus)} dokumen...")
    embeddings = model.encode(
        corpus,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    print(f"[Dense] Embedding selesai: shape {embeddings.shape}")
    return embeddings


def cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Menghitung Cosine Similarity antara satu query embedding
    dan seluruh embedding dokumen dalam corpus.

    Rumus:
      cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

    Nilai berkisar antara -1 sampai 1:
      1.0  → sangat mirip / identik secara semantik
      0.0  → tidak berkaitan
     -1.0  → berlawanan secara semantik

    Parameter:
      query_vec : vektor embedding query, shape (384,)
      doc_vecs  : matrix embedding dokumen, shape (n_doc, 384)

    Return:
      Array skor similarity, shape (n_doc,)
    """
    # Normalisasi query vector
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    # Normalisasi setiap dokumen vector
    doc_norms  = doc_vecs  / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    # Dot product = cosine similarity (karena sudah dinormalisasi)
    scores = np.dot(doc_norms, query_norm)
    return scores


# ─────────────────────────────────────────────
# 4. HYBRID SEARCH (BM25 + DENSE)
# ─────────────────────────────────────────────

def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    dense_embeddings: np.ndarray,
    model: SentenceTransformer,
    df: pd.DataFrame,
    top_k: int = 10,
    alpha: float = 0.5
) -> pd.DataFrame:
    """
    Hybrid Search: Menggabungkan BM25 (lexical) dan Dense Retrieval (semantic).

    Alur proses:
      1. Tokenisasi query → hitung BM25 score untuk semua dokumen
      2. Encode query ke vektor → hitung cosine similarity semua dokumen
      3. Normalisasi kedua skor ke rentang [0, 1] agar skala setara
      4. Gabungkan dengan bobot alpha:
           hybrid_score = α × bm25_norm + (1-α) × dense_norm
      5. Urutkan descending, ambil top-K dokumen terbaik

    Parameter:
      query            : teks pertanyaan/pencarian dari pengguna
      bm25             : indeks BM25 yang sudah dibangun
      dense_embeddings : matrix embedding semua dokumen (n_doc × 384)
      model            : model Sentence Transformer untuk encode query
      df               : DataFrame regulasi (untuk mengambil metadata)
      top_k            : jumlah hasil teratas yang dikembalikan
      alpha            : bobot BM25 (0.0 = full dense, 1.0 = full BM25)

    Contoh:
      alpha=0.5  → BM25 50% + Dense 50% (seimbang)
      alpha=0.7  → BM25 70% + Dense 30% (lebih keyword)
      alpha=0.3  → BM25 30% + Dense 70% (lebih semantik)
    """

    # ── LANGKAH 1: Skor BM25 (Lexical) ──
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)      # Shape: (n_doc,)

    # ── LANGKAH 2: Skor Dense (Semantic / Cosine Similarity) ──
    query_embedding = model.encode([query], convert_to_numpy=True)[0]  # Shape: (384,)
    dense_scores = cosine_similarity_matrix(query_embedding, dense_embeddings)  # Shape: (n_doc,)

    # ── LANGKAH 3: Normalisasi Min-Max → [0, 1] ──
    #    Diperlukan agar skala BM25 dan Dense setara sebelum digabung
    scaler = MinMaxScaler()
    bm25_norm  = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    dense_norm = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()

    # ── LANGKAH 4: Hitung Hybrid Score ──
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_norm

    # ── LANGKAH 5: Ambil Top-K ──
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["bm25_score"]   = bm25_scores[top_indices].round(4)
    results["dense_score"]  = dense_scores[top_indices].round(4)
    results["hybrid_score"] = hybrid_scores[top_indices].round(4)
    results["rank"]         = range(1, len(top_indices) + 1)

    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. CACHE INDEX (agar tidak rebuild setiap run)
# ─────────────────────────────────────────────

CACHE_FILE = "index_cache.pkl"

def save_index(bm25, dense_embeddings):
    """
    Simpan indeks BM25 dan embedding Dense ke file cache (.pkl).
    Tujuan: menghindari proses rebuild yang memakan waktu setiap
    kali aplikasi dijalankan ulang.
    """
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "dense": dense_embeddings}, f)
    print(f"[Cache] Indeks disimpan ke {CACHE_FILE}")

def load_index():
    """
    Muat indeks dari file cache jika tersedia.
    Mengembalikan (None, None) jika cache belum ada —
    sinyal untuk membangun indeks baru.
    """
    if os.path.exists(CACHE_FILE):
        print(f"[Cache] Memuat indeks dari {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        return cache["bm25"], cache["dense"]
    return None, None
