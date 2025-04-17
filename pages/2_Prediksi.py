import streamlit as st
import pandas as pd
import joblib
import gdown
import datetime
import plotly.graph_objects as go
import os

st.title("🔮 Prediksi Harga Komoditas")

# Download model dari Google Drive
url_model = 'https://drive.google.com/uc?id=1OKajzFzbAOuwliw8LwD8aF9_dscKhXam'
output_model = 'model_rf_harga.pkl'

# Download label encoder
url_encoder = 'https://drive.google.com/uc?id=1vmcx3F1c95ufQnQLaaWPmSzDsms3VnCr'
output_encoder = 'label_encoder_dict.pkl'

if not os.path.exists('model_rf_harga.pkl'):
    gdown.download(url_model, output_model, quiet=False)
if not os.path.exists('label_encoder_dict.pkl'):
    gdown.download(url_encoder, output_encoder, quiet=False)

# Load model dan encoder
model = joblib.load('model_rf_harga.pkl')
le_dict = joblib.load('label_encoder_dict.pkl')

provinsi_list = le_dict["Provinsi"].classes_
kota_list = le_dict["Kabupaten Kota"].classes_
pasar_list = le_dict["Nama Pasar"].classes_
variant_list = le_dict["Nama Variant"].classes_

provinsi = st.selectbox("Pilih Provinsi", provinsi_list)
kab_kota = st.selectbox("Pilih Kabupaten/Kota", kota_list)
nama_pasar = st.selectbox("Pilih Nama Pasar", pasar_list)
nama_variant = st.selectbox("Pilih Komoditas", variant_list)
tanggal = st.date_input("Tanggal", value=datetime.date.today())

if st.button("Prediksi"):
    day = tanggal.day
    month = tanggal.month
    dayofweek = tanggal.weekday()

    input_dict = {
        "Provinsi": le_dict["Provinsi"].transform([provinsi])[0],
        "Kabupaten Kota": le_dict["Kabupaten Kota"].transform([kab_kota])[0],
        "Nama Pasar": le_dict["Nama Pasar"].transform([nama_pasar])[0],
        "Nama Variant": le_dict["Nama Variant"].transform([nama_variant])[0],
        "day": day,
        "month": month,
        "dayofweek": dayofweek
    }

    input_df = pd.DataFrame([input_dict])
    pred_today = model.predict(input_df)[0]

    besok = tanggal + datetime.timedelta(days=1)
    input_dict_besok = input_dict.copy()
    input_dict_besok["day"] = besok.day
    input_dict_besok["month"] = besok.month
    input_dict_besok["dayofweek"] = besok.weekday()
    pred_tomorrow = model.predict(pd.DataFrame([input_dict_besok]))[0]

    st.markdown(f"📅 **Harga Hari Ini ({tanggal}):** Rp {pred_today:,.2f}")
    st.markdown(f"📅 **Harga Besok ({besok}):** Rp {pred_tomorrow:,.2f}")

    if pred_tomorrow > pred_today:
        st.markdown("### 🔺 Harga Diprediksi **Naik**", unsafe_allow_html=True)
    elif pred_tomorrow < pred_today:
        st.markdown("### 🔻 Harga Diprediksi **Turun**", unsafe_allow_html=True)
    else:
        st.markdown("### ➡️ Harga Diprediksi **Stabil**", unsafe_allow_html=True)

    # Bar chart perbandingan
    bar_df = pd.DataFrame({
        "Hari": ["Hari Ini", "Besok"],
        "Harga": [pred_today, pred_tomorrow]
    })

    fig = go.Figure([go.Bar(x=bar_df["Hari"], y=bar_df["Harga"], marker_color=["blue", "red"])])
    fig.update_layout(title="Perbandingan Harga", xaxis_title="Hari", yaxis_title="Harga", yaxis_tickprefix="Rp ")
    st.plotly_chart(fig, use_container_width=True)
