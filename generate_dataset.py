"""
=============================================================
  SCRIPT PERSIAPAN DATASET - SEARCH ENGINE REGULASI INDONESIA
  Sumber  : Kaggle - Indonesian Regulations Dataset
  URL     : https://www.kaggle.com/datasets/hermansugiharto/indonesian-regulations-dataset
  Author  : 220411100183 - Tugas UAS Temu Kembali Informasi
=============================================================

CARA PENGGUNAAN:
  1. Download dataset dari Kaggle (link di atas)
  2. Letakkan file CSV hasil download di folder yang sama dengan script ini
  3. Jalankan: python generate_dataset.py
  4. Script akan membersihkan dan menyiapkan file dataset_regulasi_indonesia.csv

CATATAN:
  Script ini juga menyediakan data fallback (dummy) agar aplikasi tetap
  bisa berjalan saat file Kaggle belum tersedia.
"""

import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

OUTPUT_FILE = "dataset_regulasi_indonesia.csv"

# Kolom yang wajib ada dalam dataset akhir
REQUIRED_COLUMNS = ["id", "nomor_peraturan", "judul", "jenis_peraturan", "tentang", "tahun"]

# Kolom utama pencarian (harus mengandung kalimat panjang sesuai ketentuan tugas)
SEARCH_FIELD = "tentang"


# ─────────────────────────────────────────────
# FUNGSI: LOAD DARI FILE KAGGLE
# ─────────────────────────────────────────────

def load_from_kaggle_csv() -> pd.DataFrame | None:
    """
    Mencoba memuat dataset dari file CSV Kaggle yang sudah diunduh.
    Mendeteksi file CSV secara otomatis di folder saat ini.
    """
    # Cari semua file CSV di folder ini (kecuali output kita sendiri)
    csv_files = [
        f for f in glob.glob("*.csv")
        if f != OUTPUT_FILE
    ]

    if not csv_files:
        print("[INFO] Tidak ada file CSV Kaggle ditemukan di folder ini.")
        return None

    # Gunakan file CSV pertama yang ditemukan
    source_file = csv_files[0]
    print(f"[INFO] Memuat file: {source_file}")

    try:
        df_raw = pd.read_csv(source_file, low_memory=False)
        print(f"[INFO] Berhasil memuat {len(df_raw)} baris, kolom: {list(df_raw.columns)}")
        return df_raw
    except Exception as e:
        print(f"[ERROR] Gagal membaca {source_file}: {e}")
        return None


# ─────────────────────────────────────────────
# FUNGSI: NORMALISASI KOLOM KAGGLE
# ─────────────────────────────────────────────

def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Memetakan nama kolom dari dataset Kaggle ke format standar project.

    Dataset Kaggle Indonesian Regulations memiliki kemungkinan nama kolom:
      - 'NOMOR', 'JUDUL', 'JENIS', 'TENTANG', 'TAHUN', 'STATUS', dsb.
    Fungsi ini memetakan kolom-kolom tersebut ke format yang digunakan app.
    """
    # Normalisasi nama kolom ke lowercase
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
    print(f"[INFO] Kolom setelah normalisasi: {list(df_raw.columns)}")

    # Pemetaan fleksibel: nama kolom Kaggle → nama kolom project
    COLUMN_MAP = {
        # nomor_peraturan
        "nomor": "nomor_peraturan",
        "no": "nomor_peraturan",
        "nomor_peraturan": "nomor_peraturan",
        "number": "nomor_peraturan",

        # judul
        "judul": "judul",
        "title": "judul",
        "nama": "judul",

        # jenis_peraturan
        "jenis": "jenis_peraturan",
        "jenis_peraturan": "jenis_peraturan",
        "type": "jenis_peraturan",
        "kategori": "jenis_peraturan",

        # tentang (field utama pencarian – kalimat panjang)
        "tentang": "tentang",
        "about": "tentang",
        "deskripsi": "tentang",
        "description": "tentang",
        "isi": "tentang",
        "materi": "tentang",

        # tahun
        "tahun": "tahun",
        "year": "tahun",
        "thn": "tahun",
    }

    df = pd.DataFrame()

    # Terapkan pemetaan
    for raw_col, std_col in COLUMN_MAP.items():
        if raw_col in df_raw.columns and std_col not in df.columns:
            df[std_col] = df_raw[raw_col]

    # Jika kolom 'tentang' tidak ada, coba pakai kolom teks terpanjang
    if "tentang" not in df.columns:
        text_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]
        if text_cols:
            # Pilih kolom dengan rata-rata panjang teks terbesar
            avg_len = {c: df_raw[c].dropna().astype(str).str.len().mean() for c in text_cols}
            best_col = max(avg_len, key=avg_len.get)
            df["tentang"] = df_raw[best_col]
            print(f"[INFO] Kolom 'tentang' dipetakan otomatis dari: '{best_col}'")

    # Isi kolom yang masih kosong dengan nilai default
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "id":
                pass  # akan dibuat di bawah
            elif col == "tahun":
                df["tahun"] = "N/A"
            else:
                df[col] = "N/A"

    return df


# ─────────────────────────────────────────────
# FUNGSI: DATA FALLBACK (jika Kaggle tidak tersedia)
# ─────────────────────────────────────────────

def get_fallback_data() -> pd.DataFrame:
    """
    Data fallback – 40 baris regulasi Indonesia.
    Digunakan hanya jika file Kaggle belum diunduh.
    Field 'tentang' berisi kalimat panjang sesuai syarat tugas.
    """
    print("[INFO] Menggunakan data fallback (dummy 40 regulasi).")
    data = [
        # ── Undang-Undang ──
        {"nomor_peraturan": "UU No. 1 Tahun 1946", "judul": "Kitab Undang-Undang Hukum Pidana (KUHP)", "jenis_peraturan": "Undang-Undang", "tahun": 1946,
         "tentang": "Peraturan hukum pidana yang mengatur tindak pidana dan sanksi hukum bagi warga negara Indonesia, mencakup kejahatan terhadap keamanan negara, nyawa, harta benda, kehormatan, dan kesusilaan serta pelanggaran-pelanggaran umum."},
        {"nomor_peraturan": "UU No. 39 Tahun 1999", "judul": "Hak Asasi Manusia", "jenis_peraturan": "Undang-Undang", "tahun": 1999,
         "tentang": "Pengakuan dan perlindungan hak asasi manusia sebagai hak dasar yang melekat pada setiap manusia sejak lahir, meliputi hak untuk hidup, kebebasan beragama, kebebasan berpendapat, hak atas pendidikan, serta hak atas perlakuan yang adil di muka hukum."},
        {"nomor_peraturan": "UU No. 11 Tahun 2008", "judul": "Informasi dan Transaksi Elektronik (ITE)", "jenis_peraturan": "Undang-Undang", "tahun": 2008,
         "tentang": "Pengaturan tentang informasi elektronik dan transaksi elektronik di Indonesia, mencakup larangan penyebaran konten ilegal, pencemaran nama baik di media sosial, penipuan online, akses ilegal ke sistem elektronik, serta ketentuan bukti elektronik dalam proses hukum."},
        {"nomor_peraturan": "UU No. 13 Tahun 2003", "judul": "Ketenagakerjaan", "jenis_peraturan": "Undang-Undang", "tahun": 2003,
         "tentang": "Pengaturan hubungan kerja antara pengusaha dan pekerja/buruh di Indonesia, mencakup upah minimum, jam kerja, cuti, perlindungan kerja, pemutusan hubungan kerja (PHK), pesangon, jamsostek, dan hak-hak normatif pekerja lainnya."},
        {"nomor_peraturan": "UU No. 36 Tahun 2009", "judul": "Kesehatan", "jenis_peraturan": "Undang-Undang", "tahun": 2009,
         "tentang": "Pengaturan sistem kesehatan nasional yang menjamin setiap warga negara memperoleh pelayanan kesehatan yang aman, bermutu, dan terjangkau, serta mengatur tanggung jawab pemerintah, tenaga kesehatan, fasilitas kesehatan, dan pembiayaan kesehatan melalui BPJS."},
        {"nomor_peraturan": "UU No. 20 Tahun 2003", "judul": "Sistem Pendidikan Nasional", "jenis_peraturan": "Undang-Undang", "tahun": 2003,
         "tentang": "Penetapan sistem pendidikan nasional yang mencakup pendidikan dasar 9 tahun wajib belajar, standar nasional pendidikan, akreditasi, kurikulum, pendidikan tinggi, pendidikan nonformal, serta hak dan kewajiban warga negara dalam memperoleh pendidikan yang bermutu."},
        {"nomor_peraturan": "UU No. 8 Tahun 1999", "judul": "Perlindungan Konsumen", "jenis_peraturan": "Undang-Undang", "tahun": 1999,
         "tentang": "Perlindungan hak-hak konsumen dalam transaksi jual beli barang dan jasa, mengatur kewajiban pelaku usaha untuk memberikan informasi yang benar, melarang praktik perdagangan yang menipu atau merugikan konsumen, serta membentuk Badan Penyelesaian Sengketa Konsumen (BPSK)."},
        {"nomor_peraturan": "UU No. 31 Tahun 1999", "judul": "Pemberantasan Tindak Pidana Korupsi", "jenis_peraturan": "Undang-Undang", "tahun": 1999,
         "tentang": "Pemberantasan tindak pidana korupsi di Indonesia yang mengatur perbuatan memperkaya diri secara melawan hukum dan merugikan keuangan negara, suap-menyuap, gratifikasi, penggelapan jabatan, serta pembuktian terbalik bagi pejabat yang memiliki harta tidak wajar."},
        {"nomor_peraturan": "UU No. 35 Tahun 2009", "judul": "Narkotika", "jenis_peraturan": "Undang-Undang", "tahun": 2009,
         "tentang": "Pengaturan tentang narkotika di Indonesia yang mengklasifikasikan narkotika ke dalam tiga golongan berdasarkan bahaya dan manfaatnya, menetapkan sanksi pidana bagi pengguna, pengedar, dan produsen narkotika, serta mengatur program rehabilitasi bagi pecandu."},
        {"nomor_peraturan": "UU No. 1 Tahun 1974", "judul": "Perkawinan", "jenis_peraturan": "Undang-Undang", "tahun": 1974,
         "tentang": "Pengaturan perkawinan di Indonesia yang menetapkan syarat sah perkawinan berdasarkan hukum agama dan hukum negara, batas usia perkawinan, hak dan kewajiban suami istri, harta bersama, perceraian, dan hak-hak anak yang lahir dari perkawinan."},
        {"nomor_peraturan": "UU No. 23 Tahun 2002", "judul": "Perlindungan Anak", "jenis_peraturan": "Undang-Undang", "tahun": 2002,
         "tentang": "Perlindungan anak Indonesia dari segala bentuk diskriminasi, eksploitasi, kekerasan, penelantaran, dan perlakuan salah, serta menjamin pemenuhan hak anak atas pendidikan, kesehatan, bermain, dan tumbuh kembang secara optimal sesuai harkat dan martabat kemanusiaan."},
        {"nomor_peraturan": "UU No. 40 Tahun 2004", "judul": "Sistem Jaminan Sosial Nasional (SJSN)", "jenis_peraturan": "Undang-Undang", "tahun": 2004,
         "tentang": "Pembentukan sistem jaminan sosial nasional yang memberikan perlindungan sosial bagi seluruh rakyat Indonesia melalui program jaminan kesehatan, jaminan kecelakaan kerja, jaminan hari tua, jaminan pensiun, dan jaminan kematian yang dikelola BPJS."},
        # ── Peraturan Pemerintah ──
        {"nomor_peraturan": "PP No. 78 Tahun 2015", "judul": "Pengupahan", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2015,
         "tentang": "Ketentuan tentang pengupahan tenaga kerja meliputi penetapan upah minimum provinsi dan kabupaten/kota, formula kenaikan upah minimum berdasarkan inflasi dan pertumbuhan ekonomi, komponen upah, tunjangan, dan mekanisme penyelesaian perselisihan upah antara pengusaha dan pekerja."},
        {"nomor_peraturan": "PP No. 44 Tahun 2015", "judul": "Jaminan Kecelakaan Kerja dan Jaminan Kematian", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2015,
         "tentang": "Penyelenggaraan program jaminan kecelakaan kerja dan jaminan kematian oleh BPJS Ketenagakerjaan yang memberikan perlindungan bagi pekerja terhadap risiko kecelakaan dalam hubungan kerja termasuk perjalanan dinas, serta santunan kematian bagi ahli waris pekerja peserta."},
        {"nomor_peraturan": "PP No. 46 Tahun 2015", "judul": "Jaminan Hari Tua", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2015,
         "tentang": "Penyelenggaraan program jaminan hari tua yang bertujuan menjamin peserta menerima uang tunai pada saat memasuki usia pensiun, mengalami cacat total tetap, atau meninggal dunia, dengan besaran manfaat berdasarkan akumulasi iuran dan hasil pengembangannya."},
        {"nomor_peraturan": "PP No. 82 Tahun 2012", "judul": "Penyelenggaraan Sistem dan Transaksi Elektronik", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2012,
         "tentang": "Ketentuan teknis penyelenggaraan sistem dan transaksi elektronik, kewajiban penyelenggara sistem elektronik mendaftarkan sistem kepada pemerintah, lokalisasi data (data center) di wilayah Indonesia, standar keamanan siber, perlindungan data pribadi, dan tata kelola platform digital."},
        {"nomor_peraturan": "PP No. 71 Tahun 2019", "judul": "Penyelenggaraan Sistem dan Transaksi Elektronik (Revisi)", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2019,
         "tentang": "Pembaruan ketentuan penyelenggaraan sistem elektronik dan transaksi elektronik dengan penekanan pada kewajiban penyelenggara sistem elektronik privat mendaftar, tata cara penanganan konten ilegal, mekanisme pemutusan akses, dan perlindungan data pribadi pengguna layanan digital."},
        # ── Peraturan Presiden ──
        {"nomor_peraturan": "Perpres No. 95 Tahun 2018", "judul": "Sistem Pemerintahan Berbasis Elektronik (SPBE)", "jenis_peraturan": "Peraturan Presiden", "tahun": 2018,
         "tentang": "Penerapan sistem pemerintahan berbasis elektronik secara terpadu di lingkungan instansi pemerintah pusat dan daerah dengan tujuan meningkatkan efisiensi, efektivitas, transparansi, dan akuntabilitas pelayanan publik melalui pemanfaatan teknologi informasi dan komunikasi."},
        {"nomor_peraturan": "Perpres No. 39 Tahun 2019", "judul": "Satu Data Indonesia", "jenis_peraturan": "Peraturan Presiden", "tahun": 2019,
         "tentang": "Penyelenggaraan tata kelola data pemerintah secara terpadu dalam kerangka Satu Data Indonesia yang menjamin ketersediaan data yang akurat, mutakhir, terpadu, dapat dipertanggungjawabkan, mudah diakses, dan dibagipakaikan antar instansi pemerintah untuk mendukung perencanaan pembangunan nasional."},
        {"nomor_peraturan": "Perpres No. 18 Tahun 2020", "judul": "RPJMN 2020-2024", "jenis_peraturan": "Peraturan Presiden", "tahun": 2020,
         "tentang": "Rencana Pembangunan Jangka Menengah Nasional tahun 2020-2024 yang memuat visi Indonesia Maju, tujuh agenda pembangunan, proyek strategis nasional, target makroekonomi, indikator pembangunan manusia, program pemulihan ekonomi, transformasi digital, dan pembangunan infrastruktur prioritas nasional."},
        {"nomor_peraturan": "Perpres No. 12 Tahun 2021", "judul": "Pengadaan Barang/Jasa Pemerintah (Revisi)", "jenis_peraturan": "Peraturan Presiden", "tahun": 2021,
         "tentang": "Pembaruan ketentuan pengadaan barang dan jasa pemerintah untuk mendorong penggunaan produk dalam negeri, UMKM, dan koperasi, menyederhanakan prosedur tender, mengintegrasikan e-purchasing dan e-procurement, serta mencegah korupsi pengadaan melalui digitalisasi dan transparansi proses."},
        # ── Peraturan Menteri ──
        {"nomor_peraturan": "Permen Kominfo No. 5 Tahun 2020", "judul": "Penyelenggara Sistem Elektronik Privat", "jenis_peraturan": "Peraturan Menteri", "tahun": 2020,
         "tentang": "Kewajiban pendaftaran penyelenggara sistem elektronik lingkup privat termasuk platform media sosial, marketplace, layanan streaming, dan aplikasi digital, serta prosedur koordinasi dengan pemerintah untuk penanganan konten yang melanggar ketentuan peraturan perundang-undangan Indonesia."},
        {"nomor_peraturan": "Permen Kominfo No. 20 Tahun 2016", "judul": "Perlindungan Data Pribadi dalam Sistem Elektronik", "jenis_peraturan": "Peraturan Menteri", "tahun": 2016,
         "tentang": "Kewajiban penyelenggara sistem elektronik dalam melindungi data pribadi pengguna, mencakup perolehan persetujuan, pembatasan penggunaan, kewajiban kerahasiaan, prosedur pemuktahiran dan penghapusan data, serta mekanisme pengaduan atas penyalahgunaan data pribadi."},
        {"nomor_peraturan": "Permen Naker No. 2 Tahun 2022", "judul": "Tata Cara dan Persyaratan Pembayaran Manfaat JHT", "jenis_peraturan": "Peraturan Menteri", "tahun": 2022,
         "tentang": "Ketentuan prosedur dan persyaratan pencairan manfaat Jaminan Hari Tua (JHT) BPJS Ketenagakerjaan, termasuk persyaratan usia pensiun 56 tahun, kriteria cacat total tetap, tata cara klaim ahli waris, dan dokumen yang diperlukan untuk pengajuan manfaat JHT."},
        {"nomor_peraturan": "Permen Kesehatan No. 28 Tahun 2014", "judul": "Pedoman JKN-KIS", "jenis_peraturan": "Peraturan Menteri", "tahun": 2014,
         "tentang": "Pedoman pelaksanaan program Jaminan Kesehatan Nasional-Kartu Indonesia Sehat yang mengatur prosedur pendaftaran peserta, iuran, manfaat pelayanan kesehatan, sistem rujukan berjenjang, ketentuan fasilitas kesehatan tingkat pertama dan lanjutan, serta mekanisme klaim dan pembayaran."},
        # ── Peraturan Daerah ──
        {"nomor_peraturan": "Perda DKI Jakarta No. 2 Tahun 2020", "judul": "Penanggulangan Corona Virus Disease (COVID-19)", "jenis_peraturan": "Peraturan Daerah", "tahun": 2020,
         "tentang": "Ketentuan penanggulangan pandemi COVID-19 di wilayah DKI Jakarta meliputi Pembatasan Sosial Berskala Besar (PSBB), kewajiban penggunaan masker di ruang publik, pengaturan kapasitas transportasi umum, sanksi pelanggaran protokol kesehatan, dan mekanisme karantina wilayah."},
        {"nomor_peraturan": "Perda Jawa Timur No. 4 Tahun 2021", "judul": "Penyelenggaraan Pendidikan", "jenis_peraturan": "Peraturan Daerah", "tahun": 2021,
         "tentang": "Penyelenggaraan sistem pendidikan di Jawa Timur yang mencakup pemerataan akses pendidikan, standar mutu pendidikan daerah, pengelolaan sekolah negeri dan swasta, beasiswa bagi siswa kurang mampu, pendidikan inklusif bagi anak berkebutuhan khusus, dan digitalisasi pembelajaran."},
        {"nomor_peraturan": "Perda Surabaya No. 5 Tahun 2019", "judul": "Rencana Tata Ruang Wilayah Kota Surabaya", "jenis_peraturan": "Peraturan Daerah", "tahun": 2019,
         "tentang": "Perencanaan tata ruang wilayah Kota Surabaya tahun 2019-2039 yang mengatur zonasi peruntukan lahan untuk kawasan perumahan, perdagangan, industri, ruang terbuka hijau, infrastruktur, kawasan pesisir, dan pengembangan kota pintar (smart city) berbasis teknologi informasi."},
        # ── Keputusan/Surat Edaran ──
        {"nomor_peraturan": "SE OJK No. 14 Tahun 2021", "judul": "Restrukturisasi Kredit Dampak COVID-19", "jenis_peraturan": "Surat Edaran", "tahun": 2021,
         "tentang": "Kebijakan restrukturisasi kredit dan pembiayaan perbankan bagi debitur yang terdampak pandemi COVID-19, mencakup perpanjangan masa restrukturisasi, mekanisme pengajuan keringanan cicilan, prosedur penilaian kelayakan debitur, dan ketentuan perlakuan akuntansi bagi bank pelaksana."},
        {"nomor_peraturan": "SE BI No. 23 Tahun 2021", "judul": "Kebijakan Moneter dan Stabilitas Sistem Keuangan", "jenis_peraturan": "Surat Edaran", "tahun": 2021,
         "tentang": "Arah kebijakan moneter Bank Indonesia dalam menjaga stabilitas nilai rupiah dan mendorong pemulihan ekonomi nasional, mencakup ketentuan suku bunga acuan, rasio kecukupan modal bank, kebijakan makroprudensial, pengembangan sistem pembayaran digital, dan koordinasi kebijakan fiskal-moneter."},
        {"nomor_peraturan": "Kepmen Naker No. 349 Tahun 2019", "judul": "Jabatan yang Dapat Diduduki Tenaga Kerja Asing", "jenis_peraturan": "Keputusan Menteri", "tahun": 2019,
         "tentang": "Penetapan jabatan dan posisi kerja yang dapat diduduki oleh tenaga kerja asing di Indonesia berdasarkan kebutuhan alih teknologi dan keahlian khusus, prosedur izin mempekerjakan tenaga kerja asing (IMTA), kewajiban pendampingan tenaga kerja lokal, dan jenis pekerjaan yang terlarang bagi asing."},
        # ── Regulasi Keuangan & Pajak ──
        {"nomor_peraturan": "UU No. 7 Tahun 2021", "judul": "Harmonisasi Peraturan Perpajakan (HPP)", "jenis_peraturan": "Undang-Undang", "tahun": 2021,
         "tentang": "Reformasi perpajakan nasional melalui perubahan tarif Pajak Penghasilan (PPh) orang pribadi, kenaikan tarif PPN secara bertahap, pemberlakuan pajak karbon, program pengungkapan sukarela (tax amnesty jilid 2), penguatan NIK sebagai NPWP, dan ketentuan pajak ekonomi digital."},
        {"nomor_peraturan": "PP No. 23 Tahun 2018", "judul": "Pajak Penghasilan UMKM", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2018,
         "tentang": "Ketentuan pajak penghasilan final dengan tarif 0,5 persen dari omzet bagi wajib pajak yang memiliki peredaran bruto (omzet) tertentu tidak melebihi Rp 4,8 miliar per tahun, meliputi UMKM orang pribadi, badan (kecuali CV/firma), serta jangka waktu pengenaan tarif final tersebut."},
        # ── Lingkungan & Agraria ──
        {"nomor_peraturan": "UU No. 32 Tahun 2009", "judul": "Perlindungan dan Pengelolaan Lingkungan Hidup", "jenis_peraturan": "Undang-Undang", "tahun": 2009,
         "tentang": "Perlindungan dan pengelolaan lingkungan hidup secara terpadu melalui perencanaan, pemanfaatan, pengendalian, pemeliharaan, pengawasan, dan penegakan hukum lingkungan, termasuk kewajiban Analisis Mengenai Dampak Lingkungan (AMDAL), izin lingkungan, dan sanksi bagi pencemaran atau perusakan lingkungan."},
        {"nomor_peraturan": "UU No. 5 Tahun 1960", "judul": "Pokok-Pokok Agraria (UUPA)", "jenis_peraturan": "Undang-Undang", "tahun": 1960,
         "tentang": "Pengaturan dasar hukum agraria nasional yang mencakup hak-hak atas tanah meliputi hak milik, hak guna usaha, hak guna bangunan, hak pakai, dan hak pengelolaan, asas bahwa bumi, air, dan kekayaan alam dikuasai negara untuk sebesar-besar kemakmuran rakyat."},
        # ── Keuangan Negara ──
        {"nomor_peraturan": "UU No. 17 Tahun 2003", "judul": "Keuangan Negara", "jenis_peraturan": "Undang-Undang", "tahun": 2003,
         "tentang": "Pengaturan keuangan negara yang mencakup pengelolaan Anggaran Pendapatan dan Belanja Negara (APBN), prinsip transparansi dan akuntabilitas, kewenangan Menteri Keuangan sebagai bendahara umum negara, pertanggungjawaban pelaksanaan APBN, serta audit keuangan oleh Badan Pemeriksa Keuangan (BPK)."},
        {"nomor_peraturan": "UU No. 1 Tahun 2004", "judul": "Perbendaharaan Negara", "jenis_peraturan": "Undang-Undang", "tahun": 2004,
         "tentang": "Pengaturan perbendaharaan negara meliputi pengelolaan dan pertanggungjawaban keuangan negara, rekening kas umum negara, tata cara pelaksanaan anggaran pendapatan dan belanja, penatausahaan keuangan, penyelesaian kerugian negara, dan sistem akuntansi pemerintahan berbasis akrual."},
        # ── Siber & Teknologi ──
        {"nomor_peraturan": "Perpres No. 47 Tahun 2023", "judul": "Strategi Keamanan Siber Nasional", "jenis_peraturan": "Peraturan Presiden", "tahun": 2023,
         "tentang": "Strategi keamanan siber nasional Indonesia 2023-2028 yang mencakup tata kelola keamanan siber, perlindungan infrastruktur informasi kritis nasional, penanganan insiden siber, pengembangan sumber daya manusia keamanan siber, kerja sama internasional, dan penguatan peran Badan Siber dan Sandi Negara (BSSN)."},
        {"nomor_peraturan": "UU No. 27 Tahun 2022", "judul": "Perlindungan Data Pribadi (PDP)", "jenis_peraturan": "Undang-Undang", "tahun": 2022,
         "tentang": "Perlindungan data pribadi warga negara Indonesia yang mengatur hak subjek data, kewajiban pengendali dan prosesor data pribadi, larangan pemrosesan data sensitif tanpa persetujuan, kewajiban notifikasi kebocoran data, sanksi administratif dan pidana bagi pelanggaran, serta pembentukan lembaga pengawas data pribadi."},
        {"nomor_peraturan": "UU No. 19 Tahun 2016", "judul": "Perubahan UU ITE", "jenis_peraturan": "Undang-Undang", "tahun": 2016,
         "tentang": "Perubahan UU Informasi dan Transaksi Elektronik yang mempertegas definisi konten yang melanggar kesusilaan, memperjelas unsur pencemaran nama baik di dunia maya, memperkuat perlindungan hak kekayaan intelektual digital, dan menambahkan ketentuan penyadapan yang sah oleh aparat penegak hukum."},
        {"nomor_peraturan": "PP No. 80 Tahun 2019", "judul": "Perdagangan Melalui Sistem Elektronik (e-Commerce)", "jenis_peraturan": "Peraturan Pemerintah", "tahun": 2019,
         "tentang": "Pengaturan perdagangan elektronik (e-commerce) di Indonesia meliputi kewajiban pelaku usaha perdagangan online, perlindungan konsumen digital, ketentuan marketplace dan platform perdagangan elektronik, larangan barang impor ilegal, dan kewajiban pelaku usaha luar negeri yang beroperasi di pasar Indonesia."},
    ]
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PERSIAPAN DATASET REGULASI INDONESIA")
    print("  Sumber: Kaggle - Indonesian Regulations Dataset")
    print("=" * 60)

    # 1. Coba load dari file Kaggle
    df_raw = load_from_kaggle_csv()

    if df_raw is not None:
        # 2a. Normalisasi kolom dataset Kaggle
        df = normalize_columns(df_raw)
        sumber = "Kaggle"
    else:
        # 2b. Gunakan data fallback
        df = get_fallback_data()
        sumber = "Fallback"

    # 3. Tambahkan ID
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    # 4. Bersihkan field utama pencarian
    #    Pastikan 'tentang' tidak kosong (syarat: kalimat panjang)
    df["tentang"] = df["tentang"].fillna("").astype(str).str.strip()
    df = df[df["tentang"].str.len() > 30].reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    # 5. Simpan ke CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n✅ Dataset berhasil disiapkan!")
    print(f"   Sumber       : {sumber}")
    print(f"   Total baris  : {len(df)}")
    print(f"   Output file  : {OUTPUT_FILE}")
    print(f"\n   Kolom dataset:")
    for col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        print(f"     - {col:<25} (rata-rata {avg_len:.0f} karakter)")
    print()
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
