import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import re

# LIBRARY TAMBAHAN
try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("Library belum terinstall. Mohon jalankan: pip install streamlit-option-menu")
    st.stop()

# PDF Setup
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    reportlab_available = True
except Exception:
    reportlab_available = False

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Website Pengelompokan Kesehatan", page_icon="🏥")

# --- CUSTOM CSS (TAMPILAN ADMIN & TOMBOL BOLD) ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">

<style>
    /* Menghilangkan header bawaan Streamlit */
    header {visibility: hidden;}
    
    /* Container Header Utama */
    .header-container {
        padding: 20px;
        text-align: center;
        background-color: white;
        border-bottom: 1px solid #eee;
        margin-bottom: 30px;
    }
    
    /* Judul Utama (Biru Besar) */
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: #4F8BF9;
        margin-bottom: 5px;
    }
    
    /* Sub-judul (Abu-abu) */
    .sub-title {
        font-family: sans-serif;
        font-size: 18px;
        color: #666;
        font-weight: 400;
    }

    /* Style Kartu Dashboard */
    .dashboard-card {
        padding: 20px; 
        border-radius: 10px; 
        color: white;
        margin-bottom: 5px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative; 
        overflow: hidden;   
    }
    
    /* Style Icon Latar Belakang */
    .card-icon-bg {
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 80px; 
        opacity: 0.2;    
        color: white;
        pointer-events: none;
    }
    
    /* --- WARNA BIRU GRADASI --- */
    .card-one   { background-color: #005BA1 !important; }  /* Biru Tua */
    .card-two   { background-color: #0078D4 !important; }  /* Biru Sedang */
    .card-three { background-color: #3FA9F5 !important; }  /* Biru Muda */
    
    /* Judul Kecil (Label) */
    .card-title { font-size: 14px; font-weight: bold; opacity: 0.9; margin-top: 5px; z-index: 2; }
    
    /* Angka Besar */
    .card-value { font-size: 42px; font-weight: 800; margin-bottom: 0px; line-height: 1; z-index: 2; }
    
    /* --- MODIFIKASI TOMBOL UMUM --- */
    div.stButton > button {
        background-color: #0078D4; 
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 900 !important; /* ULTRA BOLD */
        font-size: 16px !important;
        letter-spacing: 0.5px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #005a9e; 
        color: white !important;
        transform: translateY(-2px);
    }
    
    /* --- KHUSUS TOMBOL NAVIGASI DI BAWAH KARTU --- */
    div[data-testid="column"] div.stButton > button {
        margin-top: -5px; 
        border-radius: 8px;
        background-color: #e3f2fd; 
        color: #005BA1 !important; 
        border: 1px solid #005BA1;
    }
    div[data-testid="column"] div.stButton > button:hover {
        background-color: #005BA1;
        color: white !important;
    }

    .stDataFrame { border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def clean_numeric_string(val):
    if pd.isna(val): return np.nan
    val = str(val).lower().replace(',', '.')
    if 'bln' in val or 'bulan' in val:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        if nums: return float(nums[0]) / 12.0
    if 'hari' in val or 'hr' in val:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        if nums: return float(nums[0]) / 365.0 
    val = re.sub(r'(tahun|thn|th|kg|cm|orang|mmhg|/|)', '', val)
    return float(val.strip()) if val.strip() else np.nan

def get_detailed_risk_profile(age, map_val, bmi_val):
    """Analisis Risiko (Bahasa Awam & Positif untuk yang Sehat)"""
    # Tentukan Status Fisik dulu
    if bmi_val < 18.5: status_imt = 'Kurus'
    elif bmi_val <= 25.0: status_imt = 'Normal'
    else: status_imt = 'Gemuk'

    if map_val < 70: status_map = 'Rendah'
    elif map_val <= 100: status_map = 'Normal'
    else: status_map = 'Tinggi'

    # --- LOGIKA PENJELASAN ---
    if status_imt == 'Kurus':
        if status_map == 'Tinggi':
            risk = "RISIKO TINGGI (Darah Tinggi pada Badan Kurus)"
            patho = "Pembuluh darah kaku (tidak lentur) biasanya karena faktor usia lanjut atau stres tinggi, meskipun badan terlihat kurus."
            prog = "Risiko Tinggi: Pembuluh darah pecah (Stroke Hemoragik) jika tensi tidak segera diturunkan."
            score = 3
        elif status_map == 'Rendah':
            risk = "RISIKO SEDANG (Kurang Gizi & Darah Rendah)"
            patho = "Tubuh kekurangan cadangan energi dan volume darah, menyebabkan pompa jantung lemah."
            prog = "Rawan pingsan (Syncope), anemia berat, dan penurunan produktivitas."
            score = 2
        else:
            risk = "RISIKO SEDANG (Daya Tahan Tubuh Lemah)"
            patho = "Berat badan kurang menandakan cadangan energi (Glikogen/Lemak) sangat tipis."
            prog = "Sistem imun lemah: Mudah terserang penyakit infeksi (TBC, ISPA) dan lama sembuhnya."
            score = 2

    elif status_imt == 'Normal':
        if status_map == 'Tinggi':
            risk = "RISIKO TINGGI (Hipertensi Murni)"
            patho = "Tekanan darah naik akibat faktor genetik, konsumsi garam berlebih, atau pola hidup, bukan karena kegemukan."
            prog = "Bahaya Jangka Panjang: Kerusakan ginjal permanen (Gagal Ginjal) dan pembengkakan jantung."
            score = 3
        elif status_map == 'Rendah':
            # KASUS SEHAT/FIT
            risk = "RISIKO RENDAH (Kondisi Fit/Atletis)"
            patho = "Kerja jantung sangat efisien dan pembuluh darah elastis. Sering ditemukan pada orang yang rajin berolahraga."
            prog = "Sangat Baik. Risiko penyakit kardiovaskular sangat minim di masa depan."
            score = 1
        else:
            # KASUS SEHAT IDEAL
            risk = "SEHAT (Kondisi Ideal)"
            patho = "Metabolisme tubuh berjalan optimal. Keseimbangan antara asupan gizi dan aktivitas fisik terjaga dengan baik."
            prog = "Kondisi Prima. Angka harapan hidup tinggi dan risiko komplikasi penyakit sangat rendah."
            score = 1

    else: # Gemuk (Overweight/Obese)
        if status_map == 'Tinggi':
            risk = "RISIKO SANGAT TINGGI (Sindrom Metabolik)"
            patho = "Kombinasi tumpukan lemak yang menekan organ dan hipertensi yang memaksa jantung bekerja ekstrem."
            prog = "Sangat Berbahaya: Risiko serangan jantung mendadak (Koroner) atau Stroke penyumbatan."
            score = 4
        elif status_map == 'Rendah':
            risk = "RISIKO SEDANG (Obesitas - Beban Sendi)"
            patho = "Jantung masih mampu mengkompensasi, namun beban berat badan menekan tulang dan sendi."
            prog = "Waspada: Risiko pengapuran tulang (Osteoarthritis) dan Diabetes Melitus tipe 2."
            score = 2
        else:
            risk = "RISIKO SEDANG (Pre-Hipertensi / Waspada)"
            patho = "Lemak tubuh mulai mengganggu aliran darah. Ini adalah fase 'Lampu Kuning' sebelum jadi penyakit."
            prog = "Jika berat badan tidak turun, dalam 1-2 tahun berpotensi besar menjadi Hipertensi Permanen."
            score = 2

    return risk, patho, prog, score

def get_puskesmas_programs(age_val, map_val, bmi_val, risk_cat):
    """Rekomendasi Program (TEPAT SASARAN & NAMA PROGRAM RESMI)"""
    programs = []
    
    # 1. Program Kuratif/Pengobatan (Untuk Risiko Tinggi)
    if "TINGGI" in risk_cat:
        programs.append("**Program Rujuk Balik (PRB) BPJS**: Pemantauan obat rutin dan kondisi stabil di FKTP/Puskesmas.")
        programs.append("**Kunjungan Rumah (PERKESMAS)**: Pemantauan kepatuhan minum obat oleh perawat ke rumah pasien.")
        programs.append("**Diet Rendah Garam (DASH)**: Edukasi ketat pengurangan natrium dan makanan instan.")
    
    # 2. Program Manajemen Berat Badan
    if bmi_val > 25.0:
        programs.append("**Konseling Gizi (Pojok Gizi/POZI)**: Penyusunan menu diet defisit kalori yang terjangkau.")
        programs.append("**Klub Olahraga (PROLANIS)**: Mengikuti senam jantung sehat atau senam diabetes setiap minggu.")
    elif bmi_val < 18.5:
        programs.append("**Pemberian Makanan Tambahan (PMT)**: Suplementasi biskuit/susu tinggi kalori dari Puskesmas.")
        programs.append("**Skrining TBC & Kecacingan**: Cek laboratorium untuk mencari penyebab kurus.")

    # 3. Program Berbasis Usia
    if age_val > 60:
        programs.append("**Posyandu Lansia**: Skrining kemandirian dan senam vitalitas lansia.")
    elif age_val >= 15:
        programs.append("**Posbindu PTM**: Skrining rutin (Gula, Kolesterol, Asam Urat) minimal 1x sebulan.")
    
    # 4. Program Promotif (Untuk yang Sehat)
    if "SEHAT" in risk_cat or "RENDAH" in risk_cat:
        programs.append("**Gerakan Masyarakat Hidup Sehat (GERMAS)**: Kampanye makan buah/sayur dan aktivitas fisik 30 menit/hari.")
        programs.append("**Pemberdayaan Kader Kesehatan**: Dilibatkan sebagai *Role Model* atau penggerak kesehatan di desa.")
        
    return programs


# ==========================================
# 2. HEADER & SIDEBAR ADMIN
# ==========================================
st.markdown("""
<div class="header-container">
    <div class="main-title">Website Pengelompokan Profil Kesehatan Pasien</div>
    <div class="sub-title">Pengelompokan Profil Kesehatan Pasien Berdasarkan Kriteria Usia, Indeks Massa Tubuh, dan Tekanan arteri rata-rata (MAP) Menggunakan Metode K-Means Clustering</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏥 Navigasi")
    
    if "nav_menu" not in st.session_state:
        st.session_state["nav_menu"] = "Beranda"

    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Kelompokkan Pasien"], 
        icons=["house-door-fill", "list-task"],   
        default_index=0,
        key="nav_menu",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "#eee",
                "display": "flex",
                "align-items": "center"
            },
            "nav-link-selected": {"background-color": "#0078D4"},
        }
    )
    st.caption("© 2026 Skripsi Project Dara Ayudhia - 2209010086")

# ==========================================
# 3. KONTEN HALAMAN
# ==========================================

# --- HALAMAN BERANDA ---
if selected == "Beranda":
    total_data = len(st.session_state.df_valid) if 'df_valid' in st.session_state and st.session_state.df_valid is not None else 0
    
    if 'clustering_result' in st.session_state and st.session_state.clustering_result is not None:
        val_cluster = st.session_state.best_k 
    else:
        val_cluster = "-"
    
    def go_to_clustering():
        st.session_state["nav_menu"] = "Kelompokkan Pasien"
        
    sp_l, c1, c2, c3, sp_r = st.columns([1, 4, 4, 4, 1])
    
    with c1: 
        st.markdown(f"""<div class="dashboard-card card-one"><div class="card-icon-bg"><i class="fa-solid fa-hospital-user"></i></div><div class="card-value">{total_data}</div><div class="card-title">Data Pasien</div></div>""", unsafe_allow_html=True)
    with c2: 
        st.markdown(f"""<div class="dashboard-card card-two"><div class="card-icon-bg"><i class="fa-solid fa-clipboard-list"></i></div><div class="card-value">3</div><div class="card-title">Kriteria (Usia, IMT, MAP)</div></div>""", unsafe_allow_html=True)
    with c3: 
        st.markdown(f"""<div class="dashboard-card card-three"><div class="card-icon-bg"><i class="fa-solid fa-layer-group"></i></div><div class="card-value">{val_cluster}</div><div class="card-title">Kelompok Terbentuk</div></div>""", unsafe_allow_html=True)
    
    # --- LOGIKA TAMPILAN DINAMIS: INFO vs HASIL (BERANDA & HALAMAN HASIL) ---
    if 'clustering_result' in st.session_state and st.session_state.clustering_result is not None:
        
        st.markdown("---")
        st.markdown("### 📊 Monitoring Hasil Pengelompokan")
        
        df_res = st.session_state.clustering_result
        usia_col = st.session_state.usia_col 
        
        summary_stats = df_res.groupby('Kelompok Profil Pasien').agg({
            usia_col: ['count', 'mean'],
            'Nilai IMT': ['mean'], 
            'Nilai MAP': ['mean']
        }).reset_index()
        summary_stats.columns = ['Kelompok Profil Pasien', 'Jumlah Pasien', 'Usia Rata-rata', 'IMT Rata-rata', 'MAP Rata-rata']

        # --- LAYOUT: DIAGRAM & TABEL ---
        c_grafik, c_tabel = st.columns([1.5, 1])
        with c_grafik:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.barplot(data=summary_stats, x='Kelompok Profil Pasien', y='Jumlah Pasien', palette='viridis', ax=ax_bar)
            ax_bar.set_title("Pasien per Kelompok", fontsize=10, fontweight='bold')
            ax_bar.set_xlabel("Kelompok")
            ax_bar.set_ylabel("Jumlah")
            st.pyplot(fig_bar, use_container_width=True)
        with c_tabel:
            st.markdown("##### Tabel Ringkasan")
            display_df = summary_stats[['Kelompok Profil Pasien', 'Jumlah Pasien', 'IMT Rata-rata', 'MAP Rata-rata']]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Kotak Deskripsi (Sesuai Jumlah Cluster)
        c_box_l, c_box_c, c_box_r = st.columns([1, 8, 1])
        
        with c_box_c:
            cols = st.columns(len(summary_stats))
            
            temp_scores = []
            for _, row in summary_stats.iterrows():
                _, _, _, score = get_detailed_risk_profile(row['Usia Rata-rata'], row['MAP Rata-rata'], row['IMT Rata-rata'])
                temp_scores.append(score)
            summary_stats['Risk_Score'] = temp_scores
            summary_stats = summary_stats.sort_values('Risk_Score', ascending=False) 

            for idx, (_, row) in enumerate(summary_stats.iterrows()):
                cid = int(row['Kelompok Profil Pasien'])
                u_mean, imt_mean, map_mean = row['Usia Rata-rata'], row['IMT Rata-rata'], row['MAP Rata-rata']
                risk, patho, prog, score = get_detailed_risk_profile(u_mean, map_mean, imt_mean)
                
                # Hitung Detail Usia & Gender
                cluster_data = df_res[df_res['Kelompok Profil Pasien'] == cid]
                n_remaja = len(cluster_data[cluster_data[usia_col] < 18])
                n_lansia = len(cluster_data[cluster_data[usia_col] >= 60])
                n_dewasa = len(cluster_data) - (n_remaja + n_lansia)
                
                # Hitung Dominasi Usia
                age_counts = {"Remaja (<18 th)": n_remaja, "Dewasa (18-59 th)": n_dewasa, "Lansia (≥60 th)": n_lansia}
                dominant_age = max(age_counts, key=age_counts.get)
                
                n_pria = len(cluster_data[cluster_data['Label Jenis Kelamin'] == 1])
                n_wanita = len(cluster_data) - n_pria
                perc_pria = (n_pria / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
                perc_wanita = (n_wanita / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
                
                # Tentukan Warna Judul
                color_box = "error" if score >= 3 else "warning" if score == 2 else "success"
                
                # Tentukan Label Teks
                if score >= 2: 
                    label_sebab_res = "Penyebab (Sederhana)"
                    label_depan_res = "Bahaya ke Depan"
                else: 
                    label_sebab_res = "Kondisi Klinis"
                    label_depan_res = "Harapan ke Depan"
                
                with cols[idx]:
                    with st.container():
                        # Judul Merah/Hijau/Kuning
                        getattr(st, color_box)(f"### KELOMPOK {cid} — {risk}")
                        
                        st.metric("Total Pasien", f"{int(row['Jumlah Pasien'])} Orang")
                        st.write(f"**Dominasi Gender:** L: {perc_pria:.1f}%, P: {perc_wanita:.1f}%")
                        st.write(f"**Dominasi Usia:** {dominant_age}")
                        st.caption(f"Rincian: Remaja ({n_remaja}), Dewasa ({n_dewasa}), Lansia ({n_lansia})")
                        
                        st.markdown("---")
                        
                        # --- PERBAIKAN: SATU KOTAK BIRU MENYATU ---
                        # Kita gabung stringnya dulu agar masuk ke satu fungsi st.info()
                        full_info_text = (
                            f"**{label_sebab_res}:**\n{patho}\n\n"
                            f"**{label_depan_res}:**\n{prog}"
                        )
                        st.info(full_info_text)
                        # ------------------------------------------
    else:
        sp_info_l, c_info, sp_info_r = st.columns([1, 12, 1])
        with c_info:
            st.markdown("""
            <div style="background-color: #ebf8ff; border: 1px solid #bce3ff; color: #004085; padding: 20px; border-radius: 8px; text-align: center; margin-top: 25px;">
                <h4 style="margin-top: 0;">ℹ️ Informasi Penting</h4>
                <p style="font-size: 16px; margin-bottom: 0;">
                    👋 Selamat Datang. Silakan klik tombol di atas atau masuk ke menu <strong>Kelompokkan Pasien</strong> pada bagian Navigasi untuk mulai mengolah data pasien.
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- HALAMAN KELOMPOKKAN PASIEN ---
elif selected == "Kelompokkan Pasien":
    
    st.markdown("### 1. Unggah Data (CSV / XLSX / XLS)")
    upload = st.file_uploader("Pilih file data pasien", type=["csv", "xls", "xlsx"], key="uploader_utama")

    # Inisialisasi State
    if 'df_valid' not in st.session_state: st.session_state.df_valid = None
    if 'df_raw' not in st.session_state: st.session_state.df_raw = None
    if 'clustering_done' not in st.session_state: st.session_state.clustering_done = False
    if 'clustering_result' not in st.session_state: st.session_state.clustering_result = None
    if 'best_k' not in st.session_state: st.session_state.best_k = 3

    df_valid = None
    X_raw = None
    usia_col = None
    jk_col = None
    best_k = st.session_state.best_k

    if upload is not None:
        try:
            if upload.name.endswith('.csv'): df = pd.read_csv(upload)
            else: df = pd.read_excel(upload)
            st.session_state.df_raw = df.copy()

            st.markdown("### Data yang Diunggah")
            st.dataframe(df.assign(index=df.index+1).set_index('index'))

            tensi_cols = [c for c in df.columns if 'tensi' in c.lower() or 'tekanan' in c.lower()]
            if tensi_cols:
                split_t = df[tensi_cols[0]].astype(str).str.split('/', expand=True)
                if split_t.shape[1] == 2:
                    df['Tensi Sistolik'] = pd.to_numeric(split_t[0], errors='coerce')
                    df['Tensi Diastolik'] = pd.to_numeric(split_t[1], errors='coerce')

            cols_num = [c for c in df.columns if any(x in c.lower() for x in ['usia', 'umur', 'berat', 'tinggi'])]
            for c in cols_num:
                df[c] = pd.to_numeric(df[c].apply(clean_numeric_string), errors='coerce')

            usia_col_list = [c for c in df.columns if 'usia' in c.lower()]
            tb_col_list = [c for c in df.columns if 'tinggi' in c.lower()]
            bb_col_list = [c for c in df.columns if 'berat' in c.lower()]
            
            if not (usia_col_list and tb_col_list and bb_col_list):
                st.error("Kolom Usia, Tinggi Badan, atau Berat Badan tidak ditemukan!")
                st.stop()
                
            usia_col = usia_col_list[0]
            tb_col = tb_col_list[0]
            bb_col = bb_col_list[0]
            
            jk_cols = [c for c in df.columns if 'kelamin' in c.lower() or 'sex' in c.lower()]
            jk_col = jk_cols[0] if jk_cols else None
            
            if jk_col:
                df['Label Jenis Kelamin'] = df[jk_col].astype(str).apply(lambda x: 1 if any(k in x.lower() for k in ['laki', 'pria']) else 0)
            else:
                df['Label Jenis Kelamin'] = 0
                st.warning("Kolom Jenis Kelamin tidak ditemukan. Analisis Gender diabaikan.")

            req_cols = [usia_col, tb_col, bb_col, 'Tensi Sistolik']
            mask_invalid = df[req_cols].isna().any(axis=1) | (df[usia_col] < 1.0)
            
            if mask_invalid.any():
                st.warning(f"Ditemukan **{mask_invalid.sum()} data tidak valid**. Data ini dipisahkan dari analisis.")
                with st.expander("Lihat Data Dikecualikan"):
                    st.dataframe(df.loc[mask_invalid])

            df_valid = df[~mask_invalid].reset_index(drop=True)
            
            df_valid['Nilai IMT'] = (df_valid[bb_col] / ((df_valid[tb_col]/100)**2)).round(2)
            df_valid['Nilai MAP'] = (df_valid['Tensi Diastolik'] + (1/3)*(df_valid['Tensi Sistolik']-df_valid['Tensi Diastolik'])).round(2)
            
            st.session_state.df_valid = df_valid
            st.session_state.usia_col = usia_col
            st.session_state.jk_col = jk_col 
            
            st.success("Proses Preprocessing Selesai.")
            st.markdown("### Data Pasien Siap Diolah")
            st.dataframe(df_valid.assign(index=range(1, len(df_valid)+1)).set_index('index'))

        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    if st.session_state.df_valid is not None:
        df_valid = st.session_state.df_valid
        usia_col = st.session_state.usia_col
        jk_col = st.session_state.jk_col
        
        st.markdown("---")
        st.markdown("### 2. Evaluasi Jumlah Kelompok Optimal")

        features = [usia_col, 'Nilai IMT', 'Nilai MAP'] 
        X_raw = df_valid[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        @st.cache_data
        def run_evaluation(X):
            k_rng = range(2, 7)
            inertias = []; sils = []
            for k in k_rng:
                km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10).fit(X)
                inertias.append(km.inertia_); sils.append(silhouette_score(X, km.labels_))
            return list(k_rng), inertias, sils

        # PENENTUAN K OPTIMAL (MURNI SILHOUETTE TERTINGGI)
        K_range, inertia, sil_scores = run_evaluation(X_scaled)
        best_k_index = np.argmax(sil_scores)
        best_k = K_range[best_k_index]

        st.session_state.best_k = best_k 

        st.success(f"Sistem Merekomendasikan Pembagian Menjadi **{best_k} Kelompok** untuk Hasil Paling Optimal.")
        col_grafik, col_tabel = st.columns([1.5, 1]) 

        with col_grafik:
            fig, ax = plt.subplots(figsize=(6, 4)) 
            # Mengganti plot inertia menjadi sil_scores
            ax.plot(K_range, sil_scores, marker='o', linewidth=2, color='tab:green')
            
            # Mencari titik koordinat untuk kotak Rekomendasi
            idx_best = K_range.index(best_k) 
            score_best = sil_scores[idx_best]
            
            # Menyesuaikan posisi panah dan kotak anotasi untuk skala Silhouette (0 sampai 1)
            ax.annotate(f'Rekomendasi (k={best_k})', 
                         xy=(best_k, score_best), 
                         xytext=(best_k + 0.3, score_best - (max(sil_scores)*0.15)), # Disesuaikan agar tidak menabrak garis
                         arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                         bbox=dict(boxstyle="round", fc="white", ec="red"), 
                         color='red', fontweight='bold')
            
            # Mengubah Judul dan Label Sumbu
            ax.set_title("Grafik Evaluasi Silhouette Score", fontweight='bold')
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Silhouette Score")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig, use_container_width=True)

        with col_tabel:
            st.markdown("**Tabel Nilai Silhouette**")
            eval_df = pd.DataFrame({
                "Jumlah Kelompok": K_range, 
                "Silhouette Score": sil_scores
            })
            st.dataframe(eval_df.style.format({
                "Jumlah Kelompok": "{:.0f}", 
                "Silhouette Score": "{:.3f}"
            }), hide_index=True, use_container_width=True)
            
            st.markdown(f"""
            <div style="background-color: #ebf8ff; border: 1px solid #bce3ff; padding: 20px; border-radius: 10px; color: #004085; text-align: left; margin-top: 10px;">
                <h5 style="margin: 0; font-size: 14px; font-weight: bold; color: #004085;">Jumlah Profil Pasien Terbaik untuk Proses Pengelompokan:</h5>
                <h2 style="margin: 10px 0; font-size: 28px; font-weight: bold; color: #0056b3;">k = {best_k} Cluster</h2>
                <p style="margin: 0; font-size: 13px; color: #004085;">
                    Jumlah Kelompok ini memberikan pemisahan data paling tegas.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        if 'clustering_active' not in st.session_state: st.session_state.clustering_active = False
        
        st.write(f"Sistem akan melakukan pengelompokan otomatis menjadi **{best_k} kelompok** sesuai hasil evaluasi optimal.")
        klik_clustering = st.button("Proses Pengelompokan")

        if klik_clustering:
            st.session_state.clustering_active = True

        if st.session_state.clustering_active:
            km_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
            df_valid['Kelompok Profil Pasien'] = km_final.labels_ + 1
            st.session_state.clustering_result = df_valid 
            
            st.markdown("### 3. Hasil Pengelompokan")
            st.markdown("## Data Lengkap Hasil Pengelompokan")
            st.dataframe(df_valid.assign(index=range(1, len(df_valid)+1)).set_index('index'))

            summary_stats = df_valid.groupby('Kelompok Profil Pasien').agg({
                usia_col: ['count', 'mean'], 'Nilai IMT': ['mean'], 'Nilai MAP': ['mean'], 'Label Jenis Kelamin': ['mean']
            }).reset_index()
            summary_stats.columns = ['Kelompok Profil Pasien', 'Jumlah Pasien', 'Usia Rata-rata', 'IMT Rata-rata', 'MAP Rata-rata', 'Proporsi Laki-laki']
            summary_stats['% Laki-laki'] = (summary_stats['Proporsi Laki-laki'] * 100).apply(lambda x: f"{x:.1f}%")
            summary_stats['% Perempuan'] = ((1 - summary_stats['Proporsi Laki-laki']) * 100).apply(lambda x: f"{x:.1f}%")

            # --- MODIFIKASI LAYOUT: DIAGRAM & TABEL SEBELAH-SEBELAHAN DI HALAMAN HASIL ---
            c_grafik_res, c_tabel_res = st.columns([1.5, 1])
            
            with c_grafik_res:
                st.markdown("### Diagram Perbandingan")
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(data=summary_stats, x='Kelompok Profil Pasien', y='Jumlah Pasien', palette='viridis', ax=ax_bar)
                st.pyplot(fig_bar, use_container_width=True)
            
            with c_tabel_res:
                st.markdown("### Tabel Karakteristik")
                display_df = summary_stats[['Kelompok Profil Pasien', 'Jumlah Pasien', 'IMT Rata-rata', 'MAP Rata-rata']]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            # -----------------------------------------------------------------------------

            # ----------------------------------------------------
            # TABEL REFERENSI
            # ----------------------------------------------------
            st.markdown("### ℹ️ Informasi Standar Normal Pasien")
            st.info("""
            Tabel ini digunakan sebagai informasi mengenai standar medis pasien.

            **Sumber Rujukan Medis:**
            * [Rentang Indeks Massa Tubuh (IMT) — Alodokter](https://www.alodokter.com/memahami-kalkulator-berat-badan-ideal)
            * [Rentang Mean Arterial Pressure (MAP) — Halodoc](https://www.halodoc.com/artikel/ketahui-mean-arterial-pressure-untuk-mengukur-tekanan-arteri?srsltid=AfmBOoruTwaQ3IhyUkHWUSl072URoFaYPP1I3Fz-bKoEtk6H-ydjKyoC)
            """)
            ref_data = [
                {"Parameter Medis": "Indeks Massa Tubuh (IMT)", "Kategori & Nilai Rujukan": "🟡 Underweight (Kurus) < 18.5", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
                {"Parameter Medis": "", "Kategori & Nilai Rujukan": "🟢 Normal 18.5 - 25.0", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
                {"Parameter Medis": "", "Kategori & Nilai Rujukan": "🔴 Overweight (Gemuk) > 25.0", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
                
                {"Parameter Medis": "Mean Arterial Pressure (MAP)", "Kategori & Nilai Rujukan": "🟡 Hipotensi (Rendah) < 70 mmHg", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
                {"Parameter Medis": "", "Kategori & Nilai Rujukan": "🟢 Normal 70 - 100 mmHg", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
                {"Parameter Medis": "", "Kategori & Nilai Rujukan": "🔴 Hipertensi (Tinggi) > 100 mmHg", "Berlaku Untuk Usia": "Semua Usia (≥ 18 Th)"},
            ]
            st.table(pd.DataFrame(ref_data))
    # ----------------------------------------------------


            st.markdown("---")
            st.markdown("## 📊 Detail Profil Pasien & Saran Program Puskesmas")

            temp_scores = []
            for _, row in summary_stats.iterrows():
                _, _, _, score = get_detailed_risk_profile(row['Usia Rata-rata'], row['MAP Rata-rata'], row['IMT Rata-rata'])
                temp_scores.append(score)
            summary_stats['Risk_Score'] = temp_scores
            summary_stats = summary_stats.sort_values('Risk_Score', ascending=False)

            for _, row in summary_stats.iterrows():
                cid = int(row['Kelompok Profil Pasien'])
                total_p = int(row['Jumlah Pasien'])
                u_mean, imt_mean, map_mean = row['Usia Rata-rata'], row['IMT Rata-rata'], row['MAP Rata-rata']
                risk, patho, prog, score = get_detailed_risk_profile(u_mean, map_mean, imt_mean)
                programs = get_puskesmas_programs(u_mean, map_mean, imt_mean, risk)
                
                cluster_data = df_valid[df_valid['Kelompok Profil Pasien'] == cid]
                n_remaja = len(cluster_data[cluster_data[usia_col] < 18])
                n_lansia = len(cluster_data[cluster_data[usia_col] >= 60])
                n_dewasa = len(cluster_data) - (n_remaja + n_lansia)
                age_counts = {"Remaja (<18 th)": n_remaja, "Dewasa (18-59 th)": n_dewasa, "Lansia (≥60 th)": n_lansia}
                dominant_age = max(age_counts, key=age_counts.get)
                
                n_pria = len(cluster_data[cluster_data['Label Jenis Kelamin'] == 1])
                n_wanita = len(cluster_data) - n_pria
                perc_pria = (n_pria / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
                perc_wanita = (n_wanita / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
                
                label_cause = "Kondisi Klinis" if score <= 1 else "Penyebab (Sederhana)"
                label_prog = "Harapan ke Depan" if score <= 1 else "Bahaya ke Depan"
                color_box = "error" if score == 4 else "error" if score == 3 else "warning" if score == 2 else "success"

                with st.container():
                    getattr(st, color_box)(f"### KELOMPOK {cid} — {risk}")
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        st.metric("Total Pasien", f"{total_p} Orang")
                        st.write(f"**Gender:** L: {perc_pria:.1f}%, P: {perc_wanita:.1f}%")
                        st.write(f"**Dominasi Usia:** {dominant_age}")
                        # --- DETAIL USIA DISINI JUGA ---
                        st.caption(f"Rincian: Remaja ({n_remaja}), Dewasa ({n_dewasa}), Lansia ({n_lansia})")
                        # -------------------------------
                    with c2:
                        st.metric("Rata-rata IMT", f"{imt_mean:.1f}")
                        st.metric("Rata-rata MAP", f"{map_mean:.1f} mmHg")
                    with c3:
                        st.markdown(f"**{label_cause}:**\n{patho}")
                        st.markdown(f"**{label_prog}:**\n{prog}")
                    # --- LAYOUT BARU: PROGRAM DI KIRI, SUMBER DI KANAN ---
                    st.markdown("<br>", unsafe_allow_html=True) 
                    c_prog, c_sumber = st.columns([1.5, 1])
                    
                    with c_prog:
                        st.markdown("#### 📋 Saran Program Kesehatan Tepat Sasaran")
                        for p in programs: st.markdown(f"- {p.replace('**', '')}")
                        
                    with c_sumber:
                        st.markdown("#### 🔗 Sumber Literasi Medis")
                        st.info("""
                        **Penyebab & Bahaya (Hipertensi):**
                        * [WHO: Complications of Hypertension](https://www.who.int/news-room/factsheets/detail/hypertension#:~:text=Complications%20of%20uncontrolled%20hypertension,damage%2C%20leading%20to%20kidney%20failure)
                        * [AHA: Health Threats from High Blood Pressure](https://www.heart.org/en/health-topics/high-blood-pressure/health-threats-from-high-bloodpressure#:~:text=High%20blood%20pressure%20threatens%20your,allow%20plaque%20to%20build%20up)
                        * [Mayo Clinic: High Blood Pressure](https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/in-depth/high-blood-pressure/art-20045868)
                        
                        **Penyebab & Bahaya (Obesitas):**
                        * [WHO: Health Consequences of Overweight](https://www.who.int/news-room/questions-and-answers/item/obesity-health-consequences-of-being-overweight#:~:text=Being%20overweight%20or%20obese%20can,limit%20their%20intake%20of%20sugars.)
                        * [Mayo Clinic: Obesity Symptoms & Causes](https://www.mayoclinic.org/diseases-conditions/obesity/symptoms-causes/syc-20375742#:~:text=Family%20inheritance%20and%20influences,Certain%20diseases%20and%20medicines)
                        * [Diagnos: Obesitas - Penyebab & Dampak](https://diagnos.co.id/id/obesitas-penyebab-dampak-dan-solusi/)
                        
                        **Analisis Parameter Klinis (MAP & IMT):**
                        * [Halodoc: Fakta Mean Arterial Pressure (MAP)](https://www.halodoc.com/artikel/4-fakta-tentang-mean-arterial-pressure-yang-wajib-diketahui)
                        * [GE Healthcare: MAP Measurements](https://clinicalview.gehealthcare.com/article/taking-mystery-out-mapmeasurements#:~:text=MAP%20is%20calculated%20using%20the,6)
                        * [Halodoc: Cara Menghitung IMT & Status Gizi](https://www.halodoc.com/artikel/ini-cara-menghitung-imt-dan-fungsinya-dalam-status-gizi)
                        """)
                    # -----------------------------------------------------
                    with st.expander(f"Lihat Data Pasien Kelompok {cid}"):
                        st.dataframe(cluster_data.assign(No=range(1, len(cluster_data)+1)).set_index('No'))
                    st.markdown("---")

            st.markdown("### Peta Sebaran Pasien")
            fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_valid, x='Nilai IMT', y='Nilai MAP', hue='Kelompok Profil Pasien', palette='tab10', style='Label Jenis Kelamin', markers={1:'s', 0:'o'}, s=60, alpha=0.8, ax=ax_sc)
            st.pyplot(fig_sc)

            st.markdown("### 🧊 Visualisasi 3 Dimensi Sebaran Pasien")
            df_valid['Label Kelompok'] = df_valid['Kelompok Profil Pasien'].astype(str)
            df_valid['Ket Gender'] = df_valid['Label Jenis Kelamin'].map({1: 'Laki-laki', 0: 'Perempuan'})
            df_valid = df_valid.sort_values('Kelompok Profil Pasien')
            fig_3d = px.scatter_3d(df_valid, x='Nilai IMT', y='Nilai MAP', z=usia_col, color='Label Kelompok', symbol='Ket Gender', opacity=0.8, size_max=10, color_discrete_sequence=px.colors.qualitative.Bold, height=700)
            st.plotly_chart(fig_3d, use_container_width=True)

            if reportlab_available:
                def build_pdf_thesis():
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
                    styles = getSampleStyleSheet()
                    title = ParagraphStyle('T', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=14)
                    h1_sec = ParagraphStyle('H1Sec', parent=styles['Heading1'], fontSize=12, spaceBefore=15, textColor=colors.black, alignment=TA_CENTER)
                    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, textColor=colors.darkblue, spaceBefore=10)
                    body = ParagraphStyle('B', parent=styles['Normal'], fontSize=9, leading=11, alignment=TA_JUSTIFY)
                    elements = [Paragraph("LAPORAN LENGKAP ANALISIS CLUSTERING", title), Spacer(1, 20)]
                    
                    for _, row in summary_stats.iterrows():
                        cid = int(row['Kelompok Profil Pasien'])
                        risk, patho, prog, _ = get_detailed_risk_profile(row['Usia Rata-rata'], row['MAP Rata-rata'], row['IMT Rata-rata'])
                        programs = get_puskesmas_programs(row['Usia Rata-rata'], row['MAP Rata-rata'], row['IMT Rata-rata'], risk)
                        elements.append(Paragraph(f"KELOMPOK {cid}: {risk}", h2))
                        elements.append(Paragraph(f"Total: {int(row['Jumlah Pasien'])} Orang | IMT: {row['IMT Rata-rata']:.1f} | MAP: {row['MAP Rata-rata']:.1f}", body))
                        elements.append(Paragraph(f"Analisis: {patho}", body))
                        for p in programs: elements.append(Paragraph(f"- {p.replace('**', '')}", body))
                        elements.append(Spacer(1, 10))
                    
                    doc.build(elements)
                    buffer.seek(0)
                    return buffer.getvalue()

                st.download_button("💾 Download Laporan Lengkap (PDF)", build_pdf_thesis(), "Laporan_Skripsi_Clustering_Lengkap.pdf", "application/pdf")
    
    else:
        st.info("Silakan unggah data terlebih dahulu pada menu ini.")