"""
prediction.py — Skrip inferensi Sistem Deteksi Dini Dropout Jaya Jaya Institut.

Dipakai untuk menjalankan model di luar antarmuka Streamlit, misalnya sebagai
batch job terjadwal di akhir semester.

Contoh penggunaan
-----------------
# 1) Menilai satu berkas CSV berisi banyak mahasiswa
python prediction.py --input data.csv --output hasil_prediksi.csv

# 2) Menilai satu mahasiswa lewat argumen
python prediction.py --single \
    Curricular_units_1st_sem_approved=0 \
    Curricular_units_2nd_sem_approved=0 \
    Tuition_fees_up_to_date=0 Debtor=1 Age_at_enrollment=30
"""

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "model/dropout_model.joblib"
DEFAULTS_PATH = "model/defaults.json"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    defaults = json.load(open(DEFAULTS_PATH))
    return bundle, defaults


def score(frame: pd.DataFrame, bundle: dict, defaults: dict) -> np.ndarray:
    """Menghitung probabilitas dropout. Kolom yang hilang diisi nilai default."""
    features = bundle["cat"] + bundle["num"]
    frame = frame.copy()
    for col in features:
        if col not in frame.columns:
            frame[col] = defaults[col]
    return bundle["pipeline"].predict_proba(frame[features])[:, 1]


def level(p: float) -> str:
    return "Tinggi" if p >= 0.60 else ("Sedang" if p >= 0.30 else "Rendah")


def main():
    ap = argparse.ArgumentParser(description="Prediksi risiko dropout mahasiswa.")
    ap.add_argument("--input", help="Berkas CSV berisi data mahasiswa.")
    ap.add_argument("--output", default="hasil_prediksi.csv", help="Berkas CSV keluaran.")
    ap.add_argument("--sep", default=None, help="Pemisah kolom CSV (otomatis bila kosong).")
    ap.add_argument("--single", nargs="*", metavar="KEY=VALUE",
                    help="Prediksi satu mahasiswa dari pasangan kolom=nilai.")
    args = ap.parse_args()

    bundle, defaults = load_model()
    threshold = bundle["threshold"]

    # ---------- mode satu mahasiswa ----------
    if args.single is not None:
        record = dict(defaults)
        for item in args.single:
            if "=" not in item:
                sys.exit(f"Format salah: '{item}'. Gunakan KOLOM=NILAI.")
            k, v = item.split("=", 1)
            if k not in record:
                sys.exit(f"Kolom '{k}' tidak dikenali oleh model.")
            record[k] = float(v)
        p = float(score(pd.DataFrame([record]), bundle, defaults)[0])
        print(f"Skor risiko dropout : {p*100:.1f}%")
        print(f"Level risiko        : {level(p)}")
        print(f"Keputusan (thr {threshold}) : "
              f"{'BERISIKO DROPOUT' if p >= threshold else 'AMAN'}")
        return

    if not args.input:
        sys.exit("Sertakan --input <berkas.csv> atau gunakan --single.")

    # ---------- mode batch ----------
    sep = args.sep
    if sep is None:
        with open(args.input, encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        sep = ";" if head.count(";") > head.count(",") else ","

    df = pd.read_csv(args.input, sep=sep)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    probs = score(df, bundle, defaults)
    df["Skor_Risiko"] = (probs * 100).round(1)
    df["Level_Risiko"] = [level(p) for p in probs]
    df["Prediksi"] = np.where(probs >= threshold, "Berisiko Dropout", "Aman")

    df.sort_values("Skor_Risiko", ascending=False).to_csv(args.output, index=False)

    print(f"Total mahasiswa dinilai : {len(df)}")
    print(f"Risiko Tinggi           : {(df['Level_Risiko'] == 'Tinggi').sum()}")
    print(f"Risiko Sedang           : {(df['Level_Risiko'] == 'Sedang').sum()}")
    print(f"Risiko Rendah           : {(df['Level_Risiko'] == 'Rendah').sum()}")
    print(f"Hasil tersimpan di      : {args.output}")


if __name__ == "__main__":
    main()
