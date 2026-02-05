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

    # --- CLEANING DATA (LOGIKA BARU YANG LEBIH PINTAR) ---
    def bersihin_angka(nilai, is_target=False):
        val = str(nilai).strip()
        if val in ['..', 'nan', '', 'NaN']: return np.nan
        
        # Hapus koma (selalu hapus)
        val = val.replace(',', '')

        if is_target:
            # KALAU TARGET (PERSEN), JANGAN HAPUS TITIK!
            # Biarkan "45.6" tetap "45.6"
            try:
                return float(val)
            except:
                return np.nan
        else:
            # KALAU BUKAN TARGET (GDP/POPULASI), HAPUS TITIK RIBUAN
            # Ubah "20.130.200" jadi "20130200"
            val_clean = re.sub(r'\.', '', val) 
            try:
                angka = float(val_clean)
                
                # Koreksi Skala (Kalau angkanya kegedean gak masuk akal, bagi 1 Triliun)
                # Karena format CSV Worldbank kadang error jadi triliunan
                if angka > 100000000000: # Kalau lebih dari 100 Miliar (biasanya GDP/Energy error)
                    angka = angka / 1e12
                
                return angka
            except:
                return np.nan

    # Kolom-kolom
    col_pop = 'Population total'
    col_gdp = 'GDP per capita (current US$)'
    col_energy = 'Energy use (kg of oil equivalent per capita)'
    col_target = 'Renewable energy consumption (% of total final energy consumption)'
    
    # Isi nama negara
    if 'Country Name' in df.columns:
        df['Country Name'] = df['Country Name'].ffill()

    # TERAPKAN CLEANING SPESIFIK
    if col_pop in df.columns:
        df[col_pop] = df[col_pop].apply(lambda x: bersihin_angka(x, is_target=False))
    if col_gdp in df.columns:
        df[col_gdp] = df[col_gdp].apply(lambda x: bersihin_angka(x, is_target=False))
    if col_energy in df.columns:
        df[col_energy] = df[col_energy].apply(lambda x: bersihin_angka(x, is_target=False))
    
    # KHUSUS TARGET: PAKAI MODE is_target=True
    if col_target in df.columns:
        df[col_target] = df[col_target].apply(lambda x: bersihin_angka(x, is_target=True))
    
    # Hapus data kosong
    df_model = df.dropna(subset=[col_pop, col_gdp, col_energy, col_target])

    # --- Training Model ---
    X = df_model[[col_pop, col_gdp, col_energy]]
    y = df_model[col_target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model diperkuat
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

# --- A. PILIH NEGARA ---
st.sidebar.subheader("1. Pilih Negara")
list_negara = sorted(df_full['Country Name'].dropna().unique())
# Default ke Indonesia kalau ada
idx_default = list_negara.index('Indonesia') if 'Indonesia' in list_negara else 0
selected_country = st.sidebar.selectbox("Cari negara:", list_negara, index=idx_default)

# Ambil data terakhir negara tersebut
country_data = df_full[df_full['Country Name'] == selected_country].iloc[-1]

# Fungsi aman ambil nilai buat display default
def get_display_val(val, is_target=False):
    if pd.isna(val) or str(val).strip() == '..': return 0.0
    # Bersihin manual buat display sidebar (sesuai logika cleaning di atas)
    val_str = str(val).replace(',', '')
    if not is_target:
        val_str = val_str.replace('.', '')
    try:
        angka = float(val_str)
        if not is_target and angka > 100000000000: angka = angka / 1e12
        return angka
    except:
        return 0.0

# Nilai default dari data asli
def_pop = get_display_val(country_data['Population total'])
def_gdp = get_display_val(country_data['GDP per capita (current US$)'])
def_energy = get_display_val(country_data['Energy use (kg of oil equivalent per capita)'])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Ubah Indikator (Simulasi)")

# --- B. INPUT ANGKA (MANUAL) ---
pop_input = st.sidebar.number_input("👥 Populasi", min_value=0.0, value=def_pop, step=100000.0, format="%f")
gdp_input = st.sidebar.number_input("💰 GDP per Kapita (USD)", min_value=0.0, value=def_gdp, step=100.0, format="%.2f")
energy_input = st.sidebar.number_input("⚡ Energi (kg/capita)", min_value=0.0, value=def_energy, step=10.0, format="%.2f")

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("🚀 PREDIKSI SKENARIO", type="primary")

# ==========================================
# 4. DASHBOARD UTAMA
# ==========================================
st.title("🌍 Green Economy AI Simulator")
st.markdown(f"Analisis untuk **{selected_country}**.")

if btn_predict:
    # Siapkan input
    input_data = pd.DataFrame([[pop_input, gdp_input, energy_input]], 
                              columns=['Population total', 'GDP per capita (current US$)', 'Energy use (kg of oil equivalent per capita)'])
    
    # Scaling
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    raw_prediction = model.predict(input_scaled)[0]
    prediction_result = float(raw_prediction)
    # Batasi hasil 0-100
    prediction_result = max(0.0, min(100.0, prediction_result))

    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.info("📊 Hasil Prediksi")
        st.metric(label="Energi Terbarukan", value=f"{prediction_result:.2f}%")
        st.progress(prediction_result / 100)
        
        if prediction_result < 20: st.error("Status: RENDAH")
        elif prediction_result < 50: st.warning("Status: MENENGAH")
        else: st.success("Status: TINGGI")
        
        st.markdown("---")
        st.caption("Data Masukan:")
        st.write(f"Populasi: {pop_input:,.0f}")
        st.write(f"GDP: US$ {gdp_input:,.2f}")
        st.write(f"Energi: {energy_input:,.2f}")

    with col2:
        st.success(f"🤖 Analisis AI ({selected_country})")
        
        if "gsk_" not in GROQ_API_KEY:
            st.error("⚠️ API Key Groq error.")
        else:
            with st.spinner("AI sedang berpikir..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    prompt_text = f"""
                    Analisis negara {selected_country}:
                    - Populasi: {pop_input}
                    - GDP: US$ {gdp_input}
                    - Energi: {energy_input}
                    
                    Prediksi: {prediction_result:.2f}% energi terbarukan.
                    
                    Berikan:
                    1. Analisis Singkat (Kenapa angkanya segitu?)
                    2. 3 Rekomendasi Kebijakan.
                    """
                    
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_text}],
                        model="llama-3.3-70b-versatile",
                    )
                    st.markdown(chat_completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Gagal konek: {e}")
else:
    st.info("👈 Pilih negara dan klik tombol Prediksi.")