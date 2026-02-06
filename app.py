import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go  # <-- Library baru buat grafik donut
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Green Economy Predictor", page_icon="🌿", layout="wide")

# CSS Kustom buat Animasi Fade-In & Tampilan Rapi
st.markdown("""
<style>
    /* Animasi muncul pelan (Fade In) */
    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    /* Kotak penjelasan di bawah status */
    .info-box {
        background-color: #020617; 
        padding: 14px; 
        border-radius: 10px; 
        border-left: 4px solid #38bdf8; 
        margin-top: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# ⚠️ API KEY GROQ
# ------------------------------------------------------------------
GROQ_API_KEY = "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" 

# ==========================================
# 2. FUNGSI LOAD DATA & LATIH MODEL
# ==========================================
@st.cache_resource
def load_data_and_model():
    try:
        df = pd.read_csv('dataset worldbank PKM fixed.csv', sep=';')
    except FileNotFoundError:
        st.error("File CSV gak ketemu! Pastikan 'dataset worldbank PKM fixed.csv' ada di folder yang sama.")
        st.stop()

    # --- CLEANING DATA ---
    def bersihin_angka(nilai, is_target=False):
        val = str(nilai).strip()
        if val in ['..', 'nan', '', 'NaN']: return np.nan
        val = val.replace(',', '')
        if is_target:
            try: return float(val)
            except: return np.nan
        else:
            val_clean = re.sub(r'\.', '', val) 
            try:
                angka = float(val_clean)
                if angka > 100000000000: angka = angka / 1e12
                return angka
            except: return np.nan

    col_pop = 'Population total'
    col_gdp = 'GDP per capita (current US$)'
    col_energy = 'Energy use (kg of oil equivalent per capita)'
    col_target = 'Renewable energy consumption (% of total final energy consumption)'
    
    if 'Country Name' in df.columns: df['Country Name'] = df['Country Name'].ffill()

    if col_pop in df.columns: df[col_pop] = df[col_pop].apply(lambda x: bersihin_angka(x, False))
    if col_gdp in df.columns: df[col_gdp] = df[col_gdp].apply(lambda x: bersihin_angka(x, False))
    if col_energy in df.columns: df[col_energy] = df[col_energy].apply(lambda x: bersihin_angka(x, False))
    if col_target in df.columns: df[col_target] = df[col_target].apply(lambda x: bersihin_angka(x, True))
    
    df_model = df.dropna(subset=[col_pop, col_gdp, col_energy, col_target])

    X = df_model[[col_pop, col_gdp, col_energy]]
    y = df_model[col_target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=7)
    model.fit(X_scaled, y)

    return model, scaler, df, df_model

try:
    model, scaler, df_full, df_clean = load_data_and_model()
except Exception as e:
    st.error(f"Error sistem: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR (PANEL KONTROL)
# ==========================================
st.sidebar.header("🎛️ Panel Kontrol")

# --- PILIH NEGARA ---
st.sidebar.subheader("1. Pilih Negara")
list_negara = sorted(df_full['Country Name'].dropna().unique())
CUSTOM_OPTION = "--- Custom (Input Bebas) ---"
list_negara.insert(0, CUSTOM_OPTION)
idx_default = list_negara.index('Indonesia') if 'Indonesia' in list_negara else 0
selected_option = st.sidebar.selectbox("Cari negara / Mode Custom:", list_negara, index=idx_default)

# --- FUNGSI HELPER ---
def get_display_val(val, is_target=False):
    if pd.isna(val) or str(val).strip() == '..': return 0.0
    val_str = str(val).replace(',', '')
    if not is_target: val_str = val_str.replace('.', '')
    try:
        angka = float(val_str)
        if not is_target and angka > 100000000000: angka = angka / 1e12
        return angka
    except: return 0.0

if selected_option == CUSTOM_OPTION:
    def_pop, def_gdp, def_energy = 1000000.0, 1000.0, 500.0
    display_name = "Skenario Kustom"
else:
    country_data = df_full[df_full['Country Name'] == selected_option].iloc[-1]
    def_pop = get_display_val(country_data['Population total'])
    def_gdp = get_display_val(country_data['GDP per capita (current US$)'])
    def_energy = get_display_val(country_data['Energy use (kg of oil equivalent per capita)'])
    display_name = selected_option

st.sidebar.markdown("---")
st.sidebar.subheader("2. Ubah Indikator")

pop_input = st.sidebar.number_input("👥 Populasi", min_value=0.0, value=def_pop, step=100000.0, format="%f")
gdp_input = st.sidebar.number_input("💰 GDP per Kapita (USD)", min_value=0.0, value=def_gdp, step=100.0, format="%.2f")
energy_input = st.sidebar.number_input("⚡ Energy Use (kg of oil/kapita)", min_value=0.0, value=def_energy, step=10.0, format="%.2f")

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("🚀 PREDIKSI SKENARIO", type="primary")

# ==========================================
# 4. LOGIKA STATE MANAGEMENT
# ==========================================
if "prediction_state" not in st.session_state:
    st.session_state.prediction_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if btn_predict:
    # 1. Prediksi
    input_data = pd.DataFrame([[pop_input, gdp_input, energy_input]], 
                              columns=['Population total', 'GDP per capita (current US$)', 'Energy use (kg of oil equivalent per capita)'])
    input_scaled = scaler.transform(input_data)
    raw_prediction = model.predict(input_scaled)[0]
    result = max(0.0, min(100.0, float(raw_prediction)))

    # 2. Analisis Awal AI
    ai_analysis = ""
    if "gsk_" in GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            negara_text = display_name if selected_option != CUSTOM_OPTION else "sebuah Negara Hipotetis"
            prompt_text = f"""
            Analisis untuk {negara_text}:
            - Populasi: {pop_input}
            - GDP: US$ {gdp_input}
            - Energy Use: {energy_input} kg of oil equivalent per capita
            Prediksi: {result:.2f}% energi terbarukan.
            Berikan: 1. Analisis Singkat, 2. 3 Rekomendasi Kebijakan Konkret.
            """
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model="llama-3.3-70b-versatile",
            )
            ai_analysis = chat_completion.choices[0].message.content
        except Exception as e:
            ai_analysis = f"Gagal mengambil analisis AI: {e}"

    # 3. Simpan ke Session State
    st.session_state.prediction_state = {
        "result": result,
        "pop": pop_input,
        "gdp": gdp_input,
        "energy": energy_input,
        "name": display_name,
        "analysis": ai_analysis
    }
    
    # 4. Reset Chat History
    st.session_state.chat_history = []

# ==========================================
# 5. DASHBOARD UTAMA
# ==========================================
st.title("🌍 AI-Based System for Green Economy Policy Analysis")
st.markdown(f"Analisis untuk **{display_name}**.")

st.markdown("""
    <div style="background-color:#0f172a; padding:16px; border-radius:10px; border-left:5px solid #22c55e; margin-bottom:20px;">
        <b>📌 Catatan Analitis:</b> Hasil prediksi ini mengevaluasi dampak indikator ekonomi terhadap strategi transisi energi.
    </div>
    """, unsafe_allow_html=True)

# TAMPILKAN HASIL (DENGAN ANIMASI FADE-IN)
if st.session_state.prediction_state:
    data = st.session_state.prediction_state
    
    # Buka div animasi
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.info("📊 Hasil Prediksi Kuantitatif")
        
        # --- GRAFIK DONUT CHART (BARU!) ---
        # Ini ngegambarin proporsi energi terbarukan vs fosil
        fig = go.Figure(data=[go.Pie(
            values=[data['result'], 100 - data['result']],
            labels=['Energi Terbarukan', 'Energi Non-Terbarukan (Fosil)'],
            hole=0.65, # Bikin bolong tengah (Donut)
            marker=dict(colors=['#22c55e', '#334155']), # Hijau vs Abu-abu gelap
            textinfo='percent',
            textfont=dict(size=14),
            hoverinfo='label+percent'
        )])

        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1), # Legenda di bawah
            margin=dict(t=0, b=0, l=0, r=0),
            height=280, # Tinggi grafik
            paper_bgcolor='rgba(0,0,0,0)', # Transparan
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
        # -----------------------------------

        # Angka Gede di bawah grafik biar jelas
        st.metric(label="Persentase Adopsi", value=f"{data['result']:.2f}%")
        
        # Status Teks
        if data['result'] < 20: st.error("Status: RENDAH")
        elif data['result'] < 50: st.warning("Status: MENENGAH")
        else: st.success("Status: TINGGI")
        
        # --- KOTAK PENJELASAN (REQUEST LO) ---
        st.markdown("""
        <div class="info-box">
        <b>ℹ️ Penjelasan Konsep:</b><br>
        Prediksi <b>% energi terbarukan</b> menunjukkan seberapa besar energi bersih yang kemungkinan digunakan suatu negara berdasarkan kondisi ekonominya, 
        sedangkan <b>status adopsi</b> menggambarkan tingkat kesiapan negara tersebut dalam melakukan transisi menuju ekonomi hijau.
        </div>
        """, unsafe_allow_html=True)
        # -------------------------------------

    with col2:
        st.success(f"🤖 Analisis Kebijakan ({data['name']})")
        st.markdown(data['analysis'])
        
        st.markdown("---")
        st.caption("🔍 Parameter Input:")
        st.code(f"Populasi: {data['pop']:,.0f} | GDP: US$ {data['gdp']:,.2f} | Energy: {data['energy']:,.2f}")

    # Tutup div animasi
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 6. FITUR CHAT LANJUTAN
    # ==========================================
    st.markdown("---")
    st.subheader("💬 Asisten Kebijakan Interaktif")
    st.caption("Tanyakan detail lebih lanjut kepada AI mengenai hasil di atas.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Contoh: Apa rekomendasi teknologi energi untuk negara dengan GDP ini?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        context_prompt = f"""
        Anda adalah asisten ahli ekonomi hijau.
        Konteks: Negara {data['name']}, Populasi {data['pop']}, GDP {data['gdp']}, Energy Use {data['energy']}.
        Prediksi Renewable Energy: {data['result']:.2f}%.
        Pertanyaan User: {prompt}.
        Jawab ringkas, padat, solutif.
        """

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                client = Groq(api_key=GROQ_API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": context_prompt}],
                    model="llama-3.3-70b-versatile",
                )
                full_response = chat_completion.choices[0].message.content
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                message_placeholder.error(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    st.info("👈 Pilih negara (atau Mode Custom) dan klik tombol 'PREDIKSI SKENARIO' untuk melihat visualisasi & analisis.")
