"""
=============================================================
  STREAMLIT UI - SEARCH ENGINE REGULASI INDONESIA
  Framework : Streamlit
  Metode    : Hybrid Search (BM25 + Dense Retrieval)
  Dataset   : Indonesian Regulations Dataset (Kaggle)
  URL       : https://www.kaggle.com/datasets/hermansugiharto/
              indonesian-regulations-dataset
  Author    : 220411100183 - Tugas UAS Temu Kembali Informasi
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

# Import modul search engine buatan sendiri
from search_engine import (
    load_dataset,
    build_bm25_index,
    build_dense_index,
    hybrid_search,
    save_index,
    load_index,
)

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN STREAMLIT
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RegSearch – Regulasi Indonesia",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS CUSTOM – TEMA MERAH PUTIH + GELAP
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #c0392b 0%, #922b21 50%, #1a1a2e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(192,57,43,0.3);
}
.main-header h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; }
.header-icon    { font-size: 3rem; margin-bottom: 0.5rem; }

/* Kartu hasil pencarian */
.result-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-left: 5px solid #c0392b;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(192,57,43,0.15);
}
.result-rank {
    display: inline-block;
    background: #c0392b;
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.result-jenis {
    font-size: 0.78rem;
    color: #7f8c8d;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.result-nomor {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0.25rem 0;
}
.result-judul {
    font-size: 0.92rem;
    color: #922b21;
    font-style: italic;
    margin-bottom: 0.7rem;
}
.result-tentang {
    font-size: 0.94rem;
    color: #2c3e50;
    line-height: 1.75;
    text-align: justify;
}
.result-tahun {
    font-size: 0.78rem;
    color: #95a5a6;
    margin-top: 0.3rem;
}

/* Badge skor similarity */
.score-container {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-hybrid { background: #c0392b; color: white; }
.badge-bm25   { background: #2980b9; color: white; }
.badge-dense  { background: #27ae60; color: white; }

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #c0392b, #922b21);
    color: white;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number { font-size: 1.8rem; font-weight: 700; }
.stat-label  { font-size: 0.8rem; opacity: 0.85; }

/* Info box */
.info-box {
    background: #eaf4fb;
    border-left: 4px solid #2980b9;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #1a5276;
    margin-bottom: 1rem;
}

/* Tombol */
.stButton > button {
    background: linear-gradient(135deg, #c0392b, #922b21) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 2rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stProgress > div > div { background-color: #c0392b !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="header-icon">🏛️</div>
    <h1>RegSearch</h1>
    <p>Search Engine Regulasi Indonesia &nbsp;|&nbsp; Hybrid Search: BM25 + Dense Retrieval</p>
    <p style="font-size:0.82rem; opacity:0.7;">
        Dataset: Indonesian Regulations Dataset &nbsp;·&nbsp;
        <a href="https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset"
           target="_blank" style="color:#f1948a;">Kaggle ↗</a>
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR – PENGATURAN
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan Pencarian")

    st.markdown("### 🎚️ Bobot Hybrid")
    alpha = st.slider(
        "Alpha (Bobot BM25)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0.0 = Full Dense Semantik | 1.0 = Full BM25 Keyword"
    )
    st.caption(f"BM25: **{alpha:.0%}** | Dense: **{1-alpha:.0%}**")

    st.markdown("### 🔢 Jumlah Hasil")
    top_k = st.slider("Top-K Hasil", min_value=3, max_value=20, value=5)

    st.markdown("---")

    # Filter jenis peraturan
    st.markdown("### 📂 Filter Jenis Peraturan")
    filter_jenis = st.selectbox(
        "Jenis Peraturan:",
        ["Semua", "Undang-Undang", "Peraturan Pemerintah",
         "Peraturan Presiden", "Peraturan Menteri",
         "Peraturan Daerah", "Keputusan Menteri", "Surat Edaran"]
    )

    st.markdown("---")
    st.markdown("### 📖 Tentang Metode")
    st.markdown("""
    **Hybrid Search** menggabungkan:

    🔵 **BM25** *(Lexical)*
    > Mencari kecocokan kata kunci secara langsung menggunakan algoritma probabilistik BM25Okapi.

    🟢 **Dense Retrieval** *(Semantic)*
    > Menggunakan Sentence Embedding 384 dimensi untuk menangkap makna semantik antar teks.

    🔴 **Hybrid Score**
    > `S = α × BM25_norm + (1-α) × Dense_norm`
    """)

    st.markdown("---")
    st.markdown("### 📊 Formula Skor")
    st.latex(r"S_{hybrid} = \alpha \cdot \hat{S}_{BM25} + (1-\alpha) \cdot \hat{S}_{Dense}")

    st.markdown("---")
    st.markdown("### 💡 Contoh Query")
    example_queries = [
        "perlindungan data pribadi digital",
        "pajak penghasilan UMKM",
        "korupsi merugikan keuangan negara",
        "jaminan kesehatan BPJS",
        "keamanan siber nasional",
        "lingkungan hidup AMDAL",
        "pendidikan dasar wajib belajar",
        "tenaga kerja upah minimum",
        "narkotika rehabilitasi",
        "perkawinan syarat sah",
    ]
    for q in example_queries:
        if st.button(f"🔍 {q}", key=f"btn_{q}"):
            st.session_state["query_input"] = q


# ─────────────────────────────────────────────
# LOAD MODEL & DATASET (dengan cache Streamlit)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Cache model Sentence Transformer agar tidak reload tiap query.

    Model: paraphrase-multilingual-MiniLM-L12-v2
      - Mendukung 50+ bahasa termasuk Bahasa Indonesia
      - Dimensi embedding: 384
      - Ringan (~120MB), cocok untuk CPU
    """
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner=False)
def initialize_index(_model):
    """
    Inisialisasi indeks BM25 dan Dense Embedding.
    Di-cache agar tidak rebuild setiap kali ada interaksi user.

    Alur:
      1. Pastikan dataset tersedia (generate jika belum ada)
      2. Load dataset CSV
      3. Coba load cache indeks (.pkl)
      4. Jika tidak ada cache → build indeks baru → simpan cache
    """
    # Nama file dataset
    dataset_path = "dataset_regulasi_indonesia.csv"

    # Generate dataset jika belum ada
    if not os.path.exists(dataset_path):
        import subprocess
        subprocess.run(["python", "generate_dataset.py"], check=True)

    # Load dataset
    df = load_dataset(dataset_path)
    corpus = df["tentang"].tolist()  # Field utama: deskripsi panjang regulasi

    # Coba load dari cache
    bm25, dense_embeddings = load_index()

    if bm25 is None:
        # Build indeks baru dari awal
        bm25 = build_bm25_index(corpus)
        dense_embeddings = build_dense_index(corpus, _model)
        save_index(bm25, dense_embeddings)

    return df, bm25, dense_embeddings


# ─────────────────────────────────────────────
# LOADING STATE
# ─────────────────────────────────────────────

with st.spinner("🔄 Memuat model dan membangun indeks pencarian..."):
    model = load_model()
    df, bm25, dense_embeddings = initialize_index(model)

# ── Statistik dataset ──
col1, col2, col3, col4 = st.columns(4)

total_regulasi = len(df)
jenis_unik     = df["jenis_peraturan"].nunique() if "jenis_peraturan" in df.columns else "-"
tahun_min      = int(df["tahun"].min()) if "tahun" in df.columns and df["tahun"].notna().any() else "-"
tahun_max      = int(df["tahun"].max()) if "tahun" in df.columns and df["tahun"].notna().any() else "-"

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total_regulasi:,}</div>
        <div class="stat-label">Total Regulasi</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{jenis_unik}</div>
        <div class="stat-label">Jenis Peraturan</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">384</div>
        <div class="stat-label">Dimensi Embedding</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{tahun_min}–{tahun_max}</div>
        <div class="stat-label">Rentang Tahun</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT QUERY PENCARIAN
# ─────────────────────────────────────────────

default_query = st.session_state.get("query_input", "")

with st.container():
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="Cari Regulasi Indonesia",
            placeholder="Contoh: perlindungan data pribadi, upah minimum buruh, pajak UMKM, jaminan kesehatan...",
            value=default_query,
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Cari", use_container_width=True)


# ─────────────────────────────────────────────
# PROSES PENCARIAN & TAMPILKAN HASIL
# ─────────────────────────────────────────────

if search_clicked and query.strip():

    # Terapkan filter jenis peraturan jika dipilih
    df_filtered = df.copy()
    if filter_jenis != "Semua" and "jenis_peraturan" in df.columns:
        df_filtered = df[
            df["jenis_peraturan"].str.contains(filter_jenis, case=False, na=False)
        ].reset_index(drop=True)

        if df_filtered.empty:
            st.warning(f"⚠️ Tidak ada regulasi jenis '{filter_jenis}'. Menampilkan semua jenis.")
            df_filtered = df.copy()

    # Re-inisialisasi indeks untuk df_filtered jika berbeda ukuran
    if len(df_filtered) < len(df):
        corpus_filtered = df_filtered["tentang"].tolist()
        from rank_bm25 import BM25Okapi
        from search_engine import tokenize, build_dense_index
        bm25_f  = BM25Okapi([tokenize(d) for d in corpus_filtered])
        dense_f = build_dense_index(corpus_filtered, model)
    else:
        bm25_f  = bm25
        dense_f = dense_embeddings
        df_filtered = df

    st.markdown(f"### 📋 Hasil untuk: `{query}`")
    st.markdown(f"*{top_k} hasil teratas · Alpha BM25: {alpha} · Dense: {1-alpha:.1f}*"
                + (f" · Filter: **{filter_jenis}**" if filter_jenis != "Semua" else ""))
    st.markdown("---")

    # Jalankan hybrid search
    start_time = time.time()
    results = hybrid_search(
        query=query,
        bm25=bm25_f,
        dense_embeddings=dense_f,
        model=model,
        df=df_filtered,
        top_k=top_k,
        alpha=alpha
    )
    elapsed = time.time() - start_time

    st.markdown(f"⏱️ *Waktu pencarian: {elapsed:.3f} detik*")
    st.markdown("<br>", unsafe_allow_html=True)

    if results.empty:
        st.warning("Tidak ditemukan hasil yang relevan. Coba ubah kata kunci.")
    else:
        # Tampilkan setiap hasil dalam kartu
        for _, row in results.iterrows():
            hybrid_pct = int(row["hybrid_score"] * 100)
            bm25_val   = row["bm25_score"]
            dense_val  = row["dense_score"]

            nomor  = row.get("nomor_peraturan", "N/A")
            judul  = row.get("judul", "")
            jenis  = row.get("jenis_peraturan", "")
            tahun  = row.get("tahun", "")
            tentang = row.get("tentang", "")

            st.markdown(f"""
            <div class="result-card">
                <span class="result-rank">#{int(row['rank'])}</span>
                <div class="result-jenis">{jenis}</div>
                <div class="result-nomor">{nomor}</div>
                <div class="result-judul">{judul}</div>
                <div class="result-tentang">{tentang}</div>
                <div class="result-tahun">📅 Tahun: {tahun}</div>
                <div class="score-container">
                    <span class="score-badge badge-hybrid">⚡ Hybrid: {hybrid_pct}%</span>
                    <span class="score-badge badge-bm25">🔵 BM25: {bm25_val:.4f}</span>
                    <span class="score-badge badge-dense">🟢 Dense: {dense_val:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabel perbandingan skor ──
        with st.expander("📊 Tabel Perbandingan Skor Semua Hasil"):
            cols_show = ["rank", "nomor_peraturan", "jenis_peraturan",
                         "hybrid_score", "bm25_score", "dense_score"]
            cols_show = [c for c in cols_show if c in results.columns]
            display_df = results[cols_show].rename(columns={
                "rank": "Rank",
                "nomor_peraturan": "Nomor Peraturan",
                "jenis_peraturan": "Jenis",
                "hybrid_score": "Hybrid Score",
                "bm25_score":   "BM25 Score",
                "dense_score":  "Dense Score",
            })
            st.dataframe(
                display_df.style.background_gradient(
                    subset=["Hybrid Score"], cmap="Reds"
                ).format({
                    "Hybrid Score": "{:.4f}",
                    "BM25 Score":   "{:.4f}",
                    "Dense Score":  "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True
            )

        # ── Visualisasi bar chart skor ──
        with st.expander("📈 Visualisasi Perbandingan Skor"):
            label_col = "nomor_peraturan" if "nomor_peraturan" in results.columns else "id"
            chart_data = results[[label_col, "bm25_score", "dense_score", "hybrid_score"]].set_index(label_col)
            st.bar_chart(chart_data)


elif search_clicked and not query.strip():
    st.warning("⚠️ Masukkan kata kunci pencarian terlebih dahulu.")


# ─────────────────────────────────────────────
# SECTION: DATASET BROWSER
# ─────────────────────────────────────────────

with st.expander("📚 Jelajahi Dataset Regulasi Indonesia"):
    st.markdown(
        f"**Total: {len(df)} regulasi** | "
        f"Sumber: [Indonesian Regulations Dataset – Kaggle]"
        f"(https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset)"
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Filter jenis
        if "jenis_peraturan" in df.columns:
            jenis_list = ["Semua"] + sorted(df["jenis_peraturan"].dropna().unique().tolist())
            sel_jenis = st.selectbox("Filter Jenis:", jenis_list, key="browse_jenis")
        else:
            sel_jenis = "Semua"
    with col_f2:
        # Filter tahun
        if "tahun" in df.columns:
            tahun_list = ["Semua"] + sorted(df["tahun"].dropna().unique().tolist(), reverse=True)
            sel_tahun = st.selectbox("Filter Tahun:", tahun_list, key="browse_tahun")
        else:
            sel_tahun = "Semua"

    browse_df = df.copy()
    if sel_jenis != "Semua" and "jenis_peraturan" in df.columns:
        browse_df = browse_df[browse_df["jenis_peraturan"] == sel_jenis]
    if sel_tahun != "Semua" and "tahun" in df.columns:
        browse_df = browse_df[browse_df["tahun"] == sel_tahun]

    show_cols = [c for c in ["nomor_peraturan", "judul", "jenis_peraturan", "tahun", "tentang"] if c in browse_df.columns]
    st.dataframe(
        browse_df[show_cols].rename(columns={
            "nomor_peraturan": "Nomor Peraturan",
            "judul":           "Judul",
            "jenis_peraturan": "Jenis",
            "tahun":           "Tahun",
            "tentang":         "Tentang / Deskripsi",
        }),
        use_container_width=True,
        hide_index=True,
        height=350
    )


# ─────────────────────────────────────────────
# SECTION: INFO METODE
# ─────────────────────────────────────────────

with st.expander("ℹ️ Penjelasan Metode Hybrid Search"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        #### 🔵 BM25 (Lexical)
        **Best Match 25** adalah algoritma ranking berbasis probabilistik.

        **Kelebihan:**
        - Cepat dan efisien
        - Presisi tinggi untuk kata kunci eksak
        - Tidak butuh GPU

        **Kekurangan:**
        - Tidak mengerti sinonim/parafrase
        - Gagal jika kata kunci beda tapi makna sama
        """)

    with col_b:
        st.markdown("""
        #### 🟢 Dense Retrieval (Semantic)
        **Sentence Transformers** mengubah teks menjadi vektor 384 dimensi.

        **Kelebihan:**
        - Mengerti makna dan konteks kalimat
        - Baik untuk query parafrase
        - Mendukung Bahasa Indonesia

        **Kekurangan:**
        - Komputasi lebih besar dari BM25
        - Kurang presisi untuk kata kunci spesifik
        """)

    with col_c:
        st.markdown("""
        #### 🔴 Hybrid Search
        Menggabungkan BM25 dan Dense dengan bobot `alpha`.

        **Formula:**
        ```
        S = α × BM25_norm + (1-α) × Dense_norm
        ```

        **Kelebihan:**
        - Keunggulan dari kedua metode
        - Bobot alpha bisa disesuaikan
        - Akurasi lebih tinggi secara keseluruhan
        """)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#7f8c8d; font-size:0.85rem; padding:1rem 0;'>
    🏛️ <b>RegSearch</b> – Search Engine Regulasi Indonesia &nbsp;|&nbsp;
    Hybrid Search: BM25 + Dense Retrieval &nbsp;|&nbsp;
    Tugas UAS Temu Kembali Informasi &nbsp;|&nbsp; NIM: 220411100183
</div>
""", unsafe_allow_html=True)
