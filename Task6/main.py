import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import savgol_filter
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("accelerometer_data.csv", parse_dates=["timestamp"])

df["acc_total"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)

df["acc_smooth"] = savgol_filter(df["acc_total"], window_length=5, polyorder=2)

plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["acc_total"], label="Raw")
plt.plot(df["timestamp"], df["acc_smooth"], label="Smoothed", linewidth=2)
plt.title("Сумарне прискорення до та після згладжування")
plt.xlabel("Час")
plt.ylabel("Прискорення")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

df.set_index("timestamp", inplace=True)
agg = df.resample("D").agg({
    "acc_smooth": ["mean", "std", "max", "min"]
})
agg.columns = ["mean_acc", "std_acc", "max_acc", "min_acc"]
agg = agg.dropna().reset_index()

agg["label"] = np.where(agg["mean_acc"] > 0.5, "active", "passive")
print(agg)

X = agg[["mean_acc", "std_acc", "max_acc", "min_acc"]]
y = agg["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("== Звіт класифікації ==")
print(classification_report(y_test, y_pred))

print("== Матриця невідповідностей ==")
labels = ["active", "passive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Прогноз")
plt.ylabel("Факт")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
