import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Baca Data
df = pd.read_csv('insurance.csv')

# 2. Preprocessing (Ubah teks jadi angka)
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = pd.factorize(df['region'])[0]

# 3. Tentukan Fitur (X) dan Target (y)
X = df.drop(columns=['charges'])
y = df['charges']

# 4. Bagi Data (80% Latih, 20% Uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Latih Model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Prediksi
y_pred = model.predict(X_test)

# 7. Tampilkan Hasil Evaluasi
print("--- HASIL EVALUASI MODEL ---")
print(f"R-squared (Akurasi): {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 8. Tampilkan Koefisien (Pengaruh Variabel)
print("\n--- PENGARUH VARIABEL (COEFFICIENTS) ---")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Nilai Koefisien'])
print(coefficients)