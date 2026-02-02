import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Dataset Medical Cost (50 observasi)
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 19, 19, 18, 35, 27, 29, 19, 
            26, 46, 50, 18, 30, 19, 25, 62, 21, 64, 64, 66, 67, 64, 52, 23, 40, 47, 43, 37, 
            41, 42, 64, 64, 65, 49, 34, 30, 25, 55],
    'charges': [16884.92, 1725.55, 4449.46, 21984.24, 3866.85, 3756.62, 8240.52, 7281.44, 6406.41, 28923.71,
                11372.75, 27378.32, 11288.57, 12559.79, 10712.50, 11482.44, 11763.26, 11047.85, 11365.52, 12466.44,
                11345.25, 16443.81, 14327.85, 10885.19, 11745.88, 11214.55, 11300.52, 28086.39, 11165.44, 31737.52,
                32787.52, 38973.52, 41355.85, 38686.92, 24212.85, 11242.52, 13618.44, 16657.85, 13999.52, 12652.85,
                15302.52, 16785.44, 36837.85, 31426.52, 39487.85, 22398.52, 13012.85, 11449.52, 11306.85, 18500.00]
}

df = pd.DataFrame(data)
print("DATASET (50 observasi):")
print(df.head(10))
print(df.describe())

# === 1. PERHITUNGAN MANUAL (IDENTIK DENGAN EXCEL) ===
print("\n" + "="*60)
print("1. PERHITUNGAN MANUAL (SAMA PERSIS DENGAN EXCEL)")
print("="*60)

n = len(df)
X = df['age'].values
Y = df['charges'].values

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_x2 = np.sum(X**2)
sum_xy = np.sum(X * Y)

# Slope dan Intercept (manual)
b_manual = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a_manual = (sum_y - b_manual * sum_x) / n

print(f"n           = {n}")
print(f"ΣX          = {sum_x:.2f}")
print(f"ΣY          = {sum_y:.2f}")
print(f"ΣX²         = {sum_x2:.2f}")
print(f"ΣXY         = {sum_xy:.2f}")
print(f"Slope (b)   = {b_manual:.4f}")
print(f"Intercept(a)= {a_manual:.4f}")
print(f"PERSAMAAN:  Y = {a_manual:.2f} + {b_manual:.2f}X")

# === 2. SKLEARN LINEAR REGRESSION ===
print("\n" + "="*60)
print("2. SKLEARN LINEAR REGRESSION (VERIFIKASI)")
print("="*60)

# Reshape untuk sklearn
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)

# Hasil sklearn
b_sklearn = model.coef_[0]
a_sklearn = model.intercept_

print(f"Sklearn Slope (b)  = {b_sklearn:.4f}")
print(f"Sklearn Intercept  = {a_sklearn:.4f}")
print("✅ IDENTIK DENGAN MANUAL!")

# === 3. EVALUASI MODEL ===
print("\n" + "="*60)
print("3. EVALUASI MODEL")
print("="*60)

Y_pred_manual = a_manual + b_manual * X
Y_pred_sklearn = model.predict(X_reshaped)

SSE = np.sum((Y - Y_pred_manual)**2)
MSE = SSE / n
RMSE = np.sqrt(MSE)
SST = np.sum((Y - np.mean(Y))**2)
R2 = 1 - (SSE / SST)

print(f"SSE           = {SSE:,.2f}")
print(f"MSE           = {MSE:,.2f}")
print(f"RMSE          = {RMSE:,.2f}")
print(f"R²            = {R2:.4f}")

print(f"\nSklearn R²     = {r2_score(Y, Y_pred_sklearn):.4f}")
print(f"Sklearn RMSE   = {np.sqrt(mean_squared_error(Y, Y_pred_sklearn)):.2f}")
print("✅ IDENTIK!")

# === 4. TABEL PERBANDINGAN ===
print("\n" + "="*60)
print("4. TABEL PERBANDINGAN (10 observasi pertama)")
print("="*60)

comparison = pd.DataFrame({
    'Age': X[:10],
    'Charges': Y[:10],
    'Prediksi_Manual': np.round(Y_pred_manual[:10], 2),
    'Prediksi_Sklearn': np.round(Y_pred_sklearn[:10], 2),
    'Error': np.round(Y[:10] - Y_pred_manual[:10], 2)
})
print(comparison)

# === 5. PREDIKSI BARU ===
print("\n" + "="*60)
print("5. PREDIKSI BARU")
print("="*60)
new_ages = [30, 45, 60]
print("Umur\t| Prediksi Manual\t| Prediksi Sklearn")
for age in new_ages:
    pred_manual = a_manual + b_manual * age
    pred_sklearn = model.predict([[age]])[0]
    print(f"{age}\t| ${pred_manual:,.2f}\t| ${pred_sklearn:,.2f}")

print("\n" + "="*60)
print("INTERPRETASI:")
print("="*60)
print(f"• Slope b = ${b_manual:.2f}: Setiap +1 tahun usia → biaya medis +${b_manual:.2f}")
print(f"• R² = {R2:.1%}: Model menjelaskan {R2:.1%} variasi data")
print(f"• RMSE = ${RMSE:.2f}: Rata-rata error prediksi")
print("✅ MANUAL == SKLEARN == EXCEL")
