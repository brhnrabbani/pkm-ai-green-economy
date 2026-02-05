import streamlit as st
import pandas as pd
import numpy as np
import re
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Green Economy Predictor", page_icon="🌿", layout="wide")

# ------------------------------------------------------------------
# ⚠️ API KEY GROQ (PASTIKAN BENAR)
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

    # Kolom & Cleaning
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

# --- INPUT ANGKA ---
pop_input = st.sidebar.number_input("👥 Populasi", min_value=0.0, value=def_pop, step=100000.0, format="%f")
gdp_input = st.sidebar.number_input("💰 GDP per Kapita (USD)", min_value=0.0, value=def_gdp, step=100.0, format="%.2f")
energy_input = st.sidebar.number_input("⚡ Energy Use (kg of oil/kapita)", min_value=0.0, value=def_energy, step=10.0, format="%.2f")

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("🚀 PREDIKSI SKENARIO", type="primary")

# ==========================================
# 4. LOGIKA STATE MANAGEMENT (BIAR CHAT GAK HILANG)
# ==========================================
# Kita butuh ingatan (Session State) buat nyimpen hasil prediksi dan chat history

if "prediction_state" not in st.session_state:
    st.session_state.prediction_state = None # Belum ada prediksi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Chat masih kosong

# Kalau tombol ditekan, simpan data ke ingatan
if btn_predict:
    # 1. Lakukan Prediksi
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

    # 3. Simpan ke Session State (Biar gak ilang pas chatting)
    st.session_state.prediction_state = {
        "result": result,
        "pop": pop_input,
        "gdp": gdp_input,
        "energy": energy_input,
        "name": display_name,
        "analysis": ai_analysis
    }
    
    # 4. Reset Chat History (Karena ganti negara/data, chat lama harus ilang)
    st.session_state.chat_history = []

# ==========================================
# 5. DASHBOARD UTAMA (RENDER BERDASARKAN STATE)
# ==========================================
st.title("🌍 AI-Based System for Green Economy Policy Analysis")
st.markdown(f"Analisis untuk **{display_name}**.")

st.markdown("""
    <div style="background-color:#0f172a; padding:16px; border-radius:10px; border-left:5px solid #22c55e; margin-bottom:20px;">
        <b>📌 Catatan Analitis:</b> Hasil prediksi ini mengevaluasi dampak indikator ekonomi terhadap strategi transisi energi.
    </div>
    """, unsafe_allow_html=True)

# HANYA TAMPILKAN HASIL JIKA SUDAH ADA PREDIKSI DI MEMORY
if st.session_state.prediction_state:
    data = st.session_state.prediction_state
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.info("📊 Hasil Prediksi")
        st.metric(label="Energi Terbarukan", value=f"{data['result']:.2f}%")
        st.progress(data['result'] / 100)
        
        if data['result'] < 20: st.error("Status: RENDAH")
        elif data['result'] < 50: st.warning("Status: MENENGAH")
        else: st.success("Status: TINGGI")
        
        st.markdown("---")
        st.caption("Data Masukan:")
        st.write(f"Populasi: {data['pop']:,.0f}")
        st.write(f"GDP: US$ {data['gdp']:,.2f}")
        st.write(f"Energy Use: {data['energy']:,.2f}")

    with col2:
        st.success(f"🤖 Analisis Awal ({data['name']})")
        st.markdown(data['analysis'])

    # ==========================================
    # 6. FITUR CHAT LANJUTAN (SESSION STATE BASED)
    # ==========================================
    st.markdown("---")
    st.subheader("💬 Asisten Kebijakan Interaktif")
    st.caption("Ada pertanyaan lebih lanjut soal hasil di atas? Tanyakan pada AI.")

    # Tampilkan History Chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Chat User
    if prompt := st.chat_input("Contoh: Apa kebijakan fiskal terbaik untuk meningkatkan angka ini?"):
        # 1. Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # 2. Siapkan Konteks buat AI (Biar dia nyambung)
        context_prompt = f"""
        Anda adalah asisten ahli ekonomi hijau.
        Konteks saat ini:
        Negara: {data['name']}
        Populasi: {data['pop']}
        GDP: {data['gdp']}
        Energy Use: {data['energy']}
        Prediksi Renewable Energy: {data['result']:.2f}%
        
        Pertanyaan User: {prompt}
        Jawablah dengan ringkas, padat, dan berbasis data.
        """

        # 3. Panggil Groq
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
                full_response = f"Maaf, terjadi kesalahan koneksi AI: {e}"
                message_placeholder.error(full_response)
        
        # 4. Simpan jawaban AI
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    st.info("👈 Pilih negara (atau Mode Custom) dan klik tombol 'PREDIKSI SKENARIO' untuk melihat hasil dan memulai chat.")
