"""
Jaya Jaya Institut - Student Dropout Early Warning System
Prototype machine learning berbasis Streamlit.

Model: Logistic Regression, dilatih pada kohort Dropout vs Graduate (3.630 mahasiswa).
Mahasiswa berstatus Enrolled tidak dipakai untuk training - mereka adalah sasaran prediksi.

Menjalankan secara lokal:  streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Jaya Jaya Institut | Dropout Early Warning",
                   page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Referensi kode kategorikal (mengikuti dokumentasi dataset)
# ----------------------------------------------------------------------------
MARITAL = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
           5: "Facto Union", 6: "Legally Separated"}

APPMODE = {1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
           5: "1st phase - special contingent (Azores)", 7: "Holder of other higher course",
           10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
           16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
           18: "3rd phase - general contingent", 26: "Ordinance 533-A/99 (Different Plan)",
           27: "Ordinance 533-A/99 (Other Institution)", 39: "Over 23 years old",
           42: "Transfer", 43: "Change of course",
           44: "Technological specialization diploma holder",
           51: "Change of institution/course", 53: "Short cycle diploma holder",
           57: "Change of institution/course (International)"}

COURSE = {33: "Biofuel Production Technologies", 171: "Animation & Multimedia Design",
          8014: "Social Service (evening attendance)", 9003: "Agronomy",
          9070: "Communication Design", 9085: "Veterinary Nursing",
          9119: "Informatics Engineering", 9130: "Equinculture", 9147: "Management",
          9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
          9670: "Advertising & Marketing Management", 9773: "Journalism & Communication",
          9853: "Basic Education", 9991: "Management (evening attendance)"}

PREV_QUAL = {1: "Secondary education", 2: "Bachelor's degree", 3: "Degree", 4: "Master's",
             5: "Doctorate", 6: "Frequency of higher education", 9: "12th year - not completed",
             10: "11th year - not completed", 12: "Other - 11th year", 14: "10th year",
             15: "10th year - not completed", 19: "Basic education 3rd cycle",
             38: "Basic education 2nd cycle", 39: "Technological specialization course",
             40: "Higher education - degree (1st cycle)", 42: "Professional higher technical course",
             43: "Higher education - master (2nd cycle)"}

YESNO = {0: "Tidak", 1: "Ya"}
GENDER = {0: "Perempuan", 1: "Laki-laki"}
ATTEND = {0: "Kelas Malam", 1: "Reguler (Siang)"}
PAID = {0: "Menunggak", 1: "Lunas"}
INTL = {0: "Domestik", 1: "Internasional"}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("model/dropout_model.joblib")
    defaults = json.load(open("model/defaults.json"))
    return bundle, defaults


bundle, DEFAULTS = load_artifacts()
PIPE = bundle["pipeline"]
THRESHOLD = bundle["threshold"]
FEATURES = bundle["cat"] + bundle["num"]


def sel(label, mapping, default_key, help=None):
    keys = list(mapping.keys())
    idx = keys.index(default_key) if default_key in keys else 0
    return st.selectbox(label, keys, index=idx, format_func=lambda k: mapping[k], help=help)


def predict(frame: pd.DataFrame):
    """Mengembalikan probabilitas dropout untuk setiap baris."""
    for c in FEATURES:
        if c not in frame.columns:
            frame[c] = DEFAULTS[c]
    frame = frame[FEATURES]
    return PIPE.predict_proba(frame)[:, 1]


def risk_level(p):
    if p >= 0.60:
        return "TINGGI", "🔴", "#E4572E"
    if p >= 0.30:
        return "SEDANG", "🟡", "#F3A712"
    return "RENDAH", "🟢", "#2E86AB"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Jaya Jaya Institut")
    st.caption("Student Dropout Early Warning System")
    mode = st.radio("Mode prediksi", ["Prediksi Individu", "Prediksi Massal (CSV)"])
    st.divider()
    st.markdown(f"""
**Model** : {bundle['model_name']}
**Threshold** : {THRESHOLD}
**ROC-AUC** : 0.975
**Accuracy** : 0.935
**Recall (dropout)** : 0.894
""")
    st.divider()
    st.caption("Skor risiko adalah probabilitas mahasiswa berakhir **Dropout** dibanding "
               "**Graduate**. Model dilatih hanya pada mahasiswa yang telah menyelesaikan "
               "masa studinya. Gunakan sebagai alat penyaring awal, bukan keputusan final.")

st.title("Sistem Deteksi Dini Risiko Dropout Mahasiswa")

# ============================================================================
# MODE 1 : PREDIKSI INDIVIDU
# ============================================================================
if mode == "Prediksi Individu":
    st.write("Isi profil mahasiswa di bawah ini, lalu klik **Prediksi Risiko Dropout**.")
    data = {}

    t1, t2, t3, t4 = st.tabs(["👤 Profil & Pendaftaran", "💰 Kondisi Finansial",
                              "📘 Akademik Semester 1", "📗 Akademik Semester 2"])

    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            data["Course"] = sel("Program Studi", COURSE, 9500)
            data["Application_mode"] = sel("Jalur Masuk", APPMODE, 1)
            data["Application_order"] = st.number_input("Urutan Pilihan Prodi", 0, 9, 1)
            data["Daytime_evening_attendance"] = sel("Waktu Kuliah", ATTEND, 1)
        with c2:
            data["Age_at_enrollment"] = st.number_input("Usia saat Mendaftar", 17, 70, 20)
            data["Gender"] = sel("Jenis Kelamin", GENDER, 0)
            data["Marital_status"] = sel("Status Pernikahan", MARITAL, 1)
            data["International"] = sel("Mahasiswa Internasional", INTL, 0)
        with c3:
            data["Previous_qualification"] = sel("Pendidikan Sebelumnya", PREV_QUAL, 1)
            data["Previous_qualification_grade"] = st.number_input(
                "Nilai Pendidikan Sebelumnya (0-200)", 0.0, 200.0, 130.0, step=1.0)
            data["Admission_grade"] = st.number_input(
                "Nilai Ujian Masuk (0-200)", 0.0, 200.0, 127.0, step=1.0)
            data["Displaced"] = sel("Merantau / Displaced", YESNO, 1)
            data["Educational_special_needs"] = sel("Kebutuhan Khusus", YESNO, 0)

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            data["Tuition_fees_up_to_date"] = sel(
                "Status Pembayaran UKT", PAID, 1,
                help="Prediktor terkuat kedua setelah SKS lulus semester 2.")
        with c2:
            data["Debtor"] = sel("Memiliki Tunggakan / Debitur", YESNO, 0)
        with c3:
            data["Scholarship_holder"] = sel("Penerima Beasiswa", YESNO, 0)

        with st.expander("Indikator makroekonomi (opsional — isi jika tersedia)"):
            m1, m2, m3 = st.columns(3)
            data["Unemployment_rate"] = m1.number_input("Tingkat Pengangguran (%)", 0.0, 30.0,
                                                        float(DEFAULTS["Unemployment_rate"]), step=0.1)
            data["Inflation_rate"] = m2.number_input("Tingkat Inflasi (%)", -5.0, 20.0,
                                                     float(DEFAULTS["Inflation_rate"]), step=0.1)
            data["GDP"] = m3.number_input("Pertumbuhan GDP (%)", -10.0, 10.0,
                                          float(DEFAULTS["GDP"]), step=0.01)

    with t3:
        c1, c2, c3 = st.columns(3)
        data["Curricular_units_1st_sem_enrolled"] = c1.number_input("SKS/MK Diambil (Sem 1)", 0, 30, 6)
        data["Curricular_units_1st_sem_approved"] = c2.number_input("SKS/MK Lulus (Sem 1)", 0, 30, 5)
        data["Curricular_units_1st_sem_grade"] = c3.number_input("Rata-rata Nilai (Sem 1, 0-20)",
                                                                 0.0, 20.0, 12.0, step=0.1)
        c4, c5, c6 = st.columns(3)
        data["Curricular_units_1st_sem_evaluations"] = c4.number_input("Jumlah Evaluasi (Sem 1)", 0, 50, 8)
        data["Curricular_units_1st_sem_credited"] = c5.number_input("MK Transfer/Credited (Sem 1)", 0, 30, 0)
        data["Curricular_units_1st_sem_without_evaluations"] = c6.number_input(
            "MK Tanpa Evaluasi (Sem 1)", 0, 30, 0)

    with t4:
        c1, c2, c3 = st.columns(3)
        data["Curricular_units_2nd_sem_enrolled"] = c1.number_input("SKS/MK Diambil (Sem 2)", 0, 30, 6)
        data["Curricular_units_2nd_sem_approved"] = c2.number_input(
            "SKS/MK Lulus (Sem 2)", 0, 30, 5,
            help="Prediktor paling berpengaruh dalam model.")
        data["Curricular_units_2nd_sem_grade"] = c3.number_input("Rata-rata Nilai (Sem 2, 0-20)",
                                                                 0.0, 20.0, 12.0, step=0.1)
        c4, c5, c6 = st.columns(3)
        data["Curricular_units_2nd_sem_evaluations"] = c4.number_input("Jumlah Evaluasi (Sem 2)", 0, 50, 8)
        data["Curricular_units_2nd_sem_credited"] = c5.number_input("MK Transfer/Credited (Sem 2)", 0, 30, 0)
        data["Curricular_units_2nd_sem_without_evaluations"] = c6.number_input(
            "MK Tanpa Evaluasi (Sem 2)", 0, 30, 0)

    st.divider()
    if st.button("🔍 Prediksi Risiko Dropout", type="primary", use_container_width=True):
        prob = float(predict(pd.DataFrame([data]))[0])
        level, icon, color = risk_level(prob)

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Probabilitas Dropout", f"{prob*100:.1f}%")
        c2.metric("Level Risiko", f"{icon} {level}")
        c3.progress(min(prob, 1.0))

        if prob >= THRESHOLD:
            st.error(f"**Terdeteksi berisiko dropout** (skor {prob*100:.1f}% ≥ ambang "
                     f"{THRESHOLD*100:.1f}%). Mahasiswa ini direkomendasikan masuk program "
                     f"bimbingan khusus.")
        else:
            st.success(f"**Risiko dropout rendah** (skor {prob*100:.1f}% < ambang "
                       f"{THRESHOLD*100:.1f}%). Cukup dipantau secara rutin.")

        # ---- rekomendasi berbasis aturan dari faktor risiko yang teramati
        recs = []
        tot = data["Curricular_units_1st_sem_enrolled"] + data["Curricular_units_2nd_sem_enrolled"]
        apr = data["Curricular_units_1st_sem_approved"] + data["Curricular_units_2nd_sem_approved"]
        rate = (apr / tot * 100) if tot > 0 else 0
        if rate < 50:
            recs.append(f"Tingkat kelulusan mata kuliah hanya **{rate:.0f}%** — jadwalkan "
                        "remedial dan pendampingan akademik intensif.")
        if data["Tuition_fees_up_to_date"] == 0:
            recs.append("UKT **menunggak** — tawarkan skema cicilan, keringanan, atau beasiswa.")
        if data["Debtor"] == 1:
            recs.append("Berstatus **debitur** — libatkan bagian keuangan untuk restrukturisasi.")
        if data["Scholarship_holder"] == 0 and data["Tuition_fees_up_to_date"] == 0:
            recs.append("Belum menerima beasiswa padahal ada indikasi kendala finansial — "
                        "prioritaskan pada seleksi beasiswa berikutnya.")
        if data["Age_at_enrollment"] > 25:
            recs.append("Masuk pada usia di atas 25 tahun — tawarkan kelas fleksibel/malam "
                        "dan konseling manajemen waktu.")
        if rate < 75 and rate >= 50:
            recs.append(f"Tingkat kelulusan mata kuliah **{rate:.0f}%** berada di bawah ambang "
                        "aman 75% — jadwalkan konseling akademik preventif.")
        if data["Curricular_units_2nd_sem_grade"] < 10 and data["Curricular_units_2nd_sem_enrolled"] > 0:
            recs.append("Nilai semester 2 di bawah 10 — evaluasi beban SKS dan tunjuk dosen wali aktif.")
        if not recs:
            recs.append("Tidak ada faktor risiko dominan yang terdeteksi. Lanjutkan pemantauan berkala.")

        st.subheader("Rekomendasi Tindak Lanjut")
        for r in recs:
            st.markdown(f"- {r}")

# ============================================================================
# MODE 2 : PREDIKSI MASSAL
# ============================================================================
else:
    st.write("Unggah berkas CSV berisi data mahasiswa (format kolom mengikuti dataset "
             "`data.csv` Jaya Jaya Institut). Pemisah `;` maupun `,` keduanya didukung.")

    up = st.file_uploader("Pilih berkas CSV", type=["csv"])
    if up is not None:
        raw = up.getvalue().decode("utf-8", errors="ignore")
        sep = ";" if raw.split("\n")[0].count(";") > raw.split("\n")[0].count(",") else ","
        import io
        df = pd.read_csv(io.StringIO(raw), sep=sep)
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} kolom tidak ditemukan dan akan diisi nilai default: "
                       f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")

        probs = predict(df.copy())
        out = df.copy()
        out["Skor_Risiko(%)"] = (probs * 100).round(1)
        out["Level_Risiko"] = pd.cut(probs, [-0.01, 0.30, 0.60, 1.0],
                                     labels=["Rendah", "Sedang", "Tinggi"])
        out["Prediksi"] = np.where(probs >= THRESHOLD, "Berisiko Dropout", "Cenderung Graduate")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Mahasiswa", len(out))
        c2.metric("🔴 Risiko Tinggi", int((out["Level_Risiko"] == "Tinggi").sum()))
        c3.metric("🟡 Risiko Sedang", int((out["Level_Risiko"] == "Sedang").sum()))
        c4.metric("🟢 Risiko Rendah", int((out["Level_Risiko"] == "Rendah").sum()))

        st.bar_chart(out["Level_Risiko"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]))

        st.subheader("Daftar Prioritas Intervensi (skor risiko tertinggi)")
        show = ["Skor_Risiko(%)", "Level_Risiko", "Prediksi"] + \
               [c for c in ["Course", "Age_at_enrollment", "Tuition_fees_up_to_date", "Debtor",
                            "Curricular_units_1st_sem_approved",
                            "Curricular_units_2nd_sem_approved"] if c in out.columns]
        st.dataframe(out.sort_values("Skor_Risiko(%)", ascending=False)[show].head(50),
                     use_container_width=True)

        st.download_button("⬇️ Unduh Hasil Prediksi (CSV)",
                           out.to_csv(index=False).encode("utf-8"),
                           "hasil_prediksi_dropout.csv", "text/csv",
                           type="primary", use_container_width=True)
    else:
        st.info("Belum ada berkas yang diunggah.")
