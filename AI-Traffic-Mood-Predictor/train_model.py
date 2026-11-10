
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


np.random.seed(42)

n = 3000

jam = np.random.randint(0, 24, n)

hari = np.random.choice(
    ["Senin", "Selasa", "Rabu", "Kamis",
     "Jumat", "Sabtu", "Minggu"], n
)

cuaca = np.random.choice(
    ["Cerah", "Berawan", "Hujan"],
    n,
    p=[0.5, 0.3, 0.2]
)

kendaraan = np.random.randint(50, 1201, n)

data = pd.DataFrame({
    "Jam": jam,
    "Hari": hari,
    "Cuaca": cuaca,
    "Jumlah_Kendaraan": kendaraan
})


jam_sibuk = (
    ((jam >= 7) & (jam <= 9)) |
    ((jam >= 16) & (jam <= 19))
).astype(int)

hari_kerja = data["Hari"].isin(
    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
).astype(int)

pengaruh_cuaca = data["Cuaca"].map({
    "Cerah": 0,
    "Berawan": 5,
    "Hujan": 15
})

skor = (
    kendaraan / 1200 * 55
    + jam_sibuk * 15
    + hari_kerja * 8
    + pengaruh_cuaca
    + np.random.normal(0, 5, n)
)

data["Kondisi"] = np.where(
    skor < 35, "Lancar",
    np.where(skor < 60, "Sedang", "Padat")
)


X = data[[
    "Jam", "Hari", "Cuaca", "Jumlah_Kendaraan"
]]

y = data["Kondisi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


preprocessor = ColumnTransformer([
    ("kategori",
     OneHotEncoder(handle_unknown="ignore"),
     ["Hari", "Cuaca"])
], remainder="passthrough")


model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    ))
])


print("Sedang melatih AI...")
model.fit(X_train, y_train)


prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

print("\nTraining berhasil!")
print(f"Akurasi model: {akurasi * 100:.2f}%")


joblib.dump(model, "traffic_model.pkl")

print("Model berhasil disimpan!")
