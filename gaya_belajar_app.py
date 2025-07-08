import streamlit as st
import numpy as np
import joblib

# Load model Decision Tree
model = joblib.load("random_forest_gaya_belajar.pkl")

# Daftar pertanyaan
questions = [
    "Saya lebih mudah memahami melalui gambar / diagram.",
    "Saya lebih suka mendengarkan penjelasan dibanding membaca.",
    "Saya suka membuat catatan visual (gambar, mindmap).",
    "Saya suka berdiskusi dengan orang lain untuk belajar.",
    "Saya lebih paham jika ikut praktik langsung.",
    "Saya belajar lebih baik saat bergerak aktif.",
    "Saya suka menjelaskan ulang dengan kata-kata.",
    "Saya lebih suka video pembelajaran daripada membaca buku.",
    "Saya lebih suka membuat kerajinan atau model saat belajar.",
    "Saya senang belajar lewat suara (rekaman, lagu belajar)."
]

st.set_page_config(page_title="Prediksi Gaya Belajar", page_icon="🧠")
st.title("🧠 Prediksi Gaya Belajar Kamu")

st.markdown("Jawablah pertanyaan berikut dengan jujur. Nilai 0 = Tidak Setuju, 1 = Setuju")

# Form jawaban user
answers = []
for i, question in enumerate(questions):
    val = st.radio(f"{i+1}. {question}", [0, 1], horizontal=True, key=f"q{i}")
    answers.append(val)

# Tombol prediksi
if st.button("Prediksi Sekarang"):
    input_array = np.array(answers).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.success(f"✨ Gaya Belajar Kamu Adalah: **{prediction}**")

    # Deskripsi tiap gaya belajar
    deskripsi = {
        "Visual": "Kamu belajar paling efektif melalui gambar, diagram, dan visualisasi.",
        "Auditori": "Kamu belajar lebih baik dengan mendengarkan, diskusi, atau penjelasan verbal.",
        "Kinestetik": "Kamu butuh gerakan dan praktik langsung untuk memahami materi."
    }

    st.markdown(f"📝 **Deskripsi:** {deskripsi.get(prediction, 'Deskripsi tidak tersedia.')}")
