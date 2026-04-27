import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Bilerek "Kirli" bir veri seti oluşturuyoruz
data = {
    'Glukoz': [85, 120, 0, 150, 95, 0, 110, 180, 130, 90], # 0 olanlar hatalı veri (Glukoz 0 olamaz)
    'Tansiyon': [70, 80, 75, 0, 65, 85, 90, 110, 0, 70],  # 0 olanlar hatalı
    'BMI': [22.5, 28.4, 33.1, 35.2, 21.0, 26.5, 30.2, 40.1, 29.5, 24.2],
    'Yas': [25, 35, 45, 55, 22, 33, 40, 60, 38, 29],
    'Sonuc': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0] # 1: Diyabet Riski, 0: Risk Yok
}

df = pd.DataFrame(data)

# 2. VERİ TEMİZLEME (Data Cleaning) - GitHub'da en çok puan toplayacak yer burası!
# Glukoz ve Tansiyon 0 olan yerleri, o sütunun ortalamasıyla dolduralım
df['Glukoz'] = df['Glukoz'].replace(0, df['Glukoz'].mean())
df['Tansiyon'] = df['Tansiyon'].replace(0, df['Tansiyon'].mean())

print("Veri Temizlendi! Artık 0 değerimiz kalmadı.")

plt.figure(figsize=(10, 5))
# 'Sonuc' sütununu 'hue' kısmına da ekliyoruz ve legend=False diyerek gereksiz kutuyu kaldırıyoruz
sns.boxplot(data=df, x='Sonuc', y='BMI', hue='Sonuc', palette='Set2', legend=False)
plt.title('Diyabet Durumuna Göre BMI (Vücut Kitle İndeksi) Dağılımı')
plt.xlabel('Diyabet Riski (0: Yok, 1: Var)')
plt.ylabel('Vücut Kitle İndeksi (BMI)')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Veriyi Hazırlama (X: Özellikler, y: Hedef Sonuç)
X = df[['Glukoz', 'Tansiyon', 'BMI', 'Yas']]
y = df['Sonuc']

# 2. Veriyi Eğitim ve Test olarak ikiye bölüyoruz (%80 öğrenme, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Oluşturma (Rastgele Orman Algoritması)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Tahmin Yapma
tahminler = model.predict(X_test)
basari = accuracy_score(y_test, tahminler)

print(f"Modelin Başarı Oranı: %{basari * 100}")