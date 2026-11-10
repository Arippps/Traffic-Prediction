
import streamlit as st
import pandas as pd
import joblib


model = joblib.load("traffic_model.pkl")


st.set_page_config(
    page_title="Traffic Mood Predictor",
    page_icon="🚦",
    layout="wide"
)


st.title("🚦 AI Traffic Mood Predictor")
st.caption("Artificial Intelligence Based Traffic Prediction System")

st.markdown("---")


st.sidebar.header("Traffic Parameters")

jam = st.sidebar.slider(
    "Jam perjalanan",
    0, 23, 8
)

hari = st.sidebar.selectbox(
    "Hari",
    ["Senin", "Selasa", "Rabu", "Kamis",
     "Jumat", "Sabtu", "Minggu"]
)

cuaca = st.sidebar.selectbox(
    "Kondisi cuaca",
    ["Cerah", "Berawan", "Hujan"]
)

kendaraan = st.sidebar.slider(
    "Jumlah kendaraan",
    50, 1200, 500, step=50
)


input_data = pd.DataFrame([{
    "Jam": jam,
    "Hari": hari,
    "Cuaca": cuaca,
    "Jumlah_Kendaraan": kendaraan
}])


prediksi = model.predict(input_data)[0]
probabilitas = model.predict_proba(input_data)[0]

kelas = model.classes_


warna = {
    "Lancar": "🟢",
    "Sedang": "🟡",
    "Padat": "🔴"
}


st.subheader("Traffic Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Waktu", f"{jam:02d}:00")

with col2:
    st.metric("Jumlah Kendaraan", f"{kendaraan:,}")

with col3:
    st.metric("Kondisi Cuaca", cuaca)

st.markdown("### Predicted Traffic Condition")

st.success(
    f"{warna[prediksi]} Kondisi lalu lintas diprediksi: {prediksi.upper()}"
)


if prediksi == "Lancar":
    st.write("Jalan diperkirakan lancar. Perjalanan dapat dilakukan seperti biasa.")
elif prediksi == "Sedang":
    st.warning("Kepadatan mulai meningkat. Siapkan waktu perjalanan tambahan.")
else:
    st.error("Kepadatan tinggi diprediksi. Pertimbangkan rute alternatif.")


st.markdown("---")
st.subheader("Prediction Probability")

prob_df = pd.DataFrame({
    "Kondisi": kelas,
    "Probabilitas (%)": probabilitas * 100
})

st.bar_chart(
    prob_df.set_index("Kondisi")
)

st.dataframe(
    prob_df.style.format({
        "Probabilitas (%)": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)


st.markdown("---")
st.subheader("Input Data Summary")
st.dataframe(input_data, use_container_width=True)

st.caption(
    "AI Traffic Mood Predictor | Random Forest Machine Learning"
)
