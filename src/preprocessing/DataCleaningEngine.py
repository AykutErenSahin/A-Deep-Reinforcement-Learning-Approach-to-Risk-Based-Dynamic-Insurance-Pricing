# Gerekli kütüphaneleri içe aktarın
import pandas as pd
import numpy as np 

# Veri setini okuyun
# Dosya adının Colab ortamınızda tam olarak bu isimle bulunduğundan emin olun.
df = pd.read_csv('RawClinicalData.csv')

# Veri setinin genel durumunu kontrol edin
print("İlk 5 satır:")
print(df.head())
print("\nVeri tipleri ve eksik değer özet bilgisi:")
df.info()

# Veri Sızıntısına (Data Leakage) neden olabilecek veya çok fazla eksik değere sahip sütunlar
columns_to_drop = [
    # Gelecek Tahminleri (Data Leakage Riski)
    'surv2m', 'surv6m', 'prg2m', 'prg6m',
    # Çok Fazla Eksik Veya Fazla Detay İçeren Maliyet/İdari Sütunlar
    'charges', 'totcst', 'totmcst', 'scoma',
    # İleri Düzey Fonksiyonel Durum Sütunları
    'adlp', 'adls', 'sfdm2', 'adlsc',
    # Tarih/Gün Bazlı Detaylar
    'hday', 'dnrday'
]

df.drop(columns=columns_to_drop, inplace=True)

print("\nSilme sonrası sütunlar:")
print(df.columns)

# 'd.time' sütununda NaN değerler olabilir. Bunları 0 ile doldurmak, takibin olmadığı anlamına gelir.
# d.time'ı, hastaneye yatıştan itibaren geçen gün sayısı olarak varsayıyoruz (bu tip veri setlerinde yaygın bir kullanım).

# d.time'daki eksik değerleri (NaN) 0 ile doldurun (Takip süresi yok)
df['d.time'].fillna(0, inplace=True)

# Ömrü hesaplama (Yaş + Takip Süresi/365.25)
# Kişinin tahmini ölüm yaşı/yaşadığı süre:
df['death_age'] = df['age'] + (df['d.time'] / 365.25)

# Survival Analizinde kullanacağımız iki değişken:
df['Event'] = df['death']  # Olay (1=Öldü, 0=Hayatta/Sansürlü)
df['Time'] = df['death_age'] # Süre (Yaşadığı toplam süre)

# Artık 'age', 'd.time', 'death' sütunlarını modelden çıkarabiliriz.
df.drop(columns=['age', 'd.time', 'death'], inplace=True)

print("\nYeni Hedef Değişkenler:")
print(df[['Time', 'Event']].head())

# Cinsiyet: male=1, female=0
df['sex'] = df['sex'].replace({'male': 1, 'female': 0})

# DNR (Do Not Resuscitate): yes dnr=1, no dnr=0
df['dnr'] = df['dnr'].replace({'yes dnr': 1, 'no dnr': 0})

# Hastalık Durumları (ca, diabetes, dementia) genellikle 0/1/metin şeklinde olabilir.
# Eğer metinsel değerler varsa, bunları da sayısallaştırmak gerekir (Örn: metastatic, localized gibi)
# Bu veri setinde 'ca' sütununda 'no', 'yes' ve alt kategoriler var gibi duruyor, bu yüzden bunu B şıkkında ele alalım.

# One-Hot Encoding uygulanacak sütunlar
categorical_cols = ['race', 'income', 'dzgroup', 'dzclass', 'ca']

# pd.get_dummies() fonksiyonu ile dönüştürme yapın
# drop_first=True: İlk kategoriyi atarak modeldeki çoklu doğrusal ilişki (multicollinearity) sorununu azaltırız.
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nEncoding sonrası sütun sayısı:", df.shape[1])
print("Encoded sütunlardan bazıları:")
print([col for col in df.columns if 'race_' in col or 'dzgroup_' in col][:5])

# Not: Bu hesaplamalar, veriyi henüz ölçeklendirmediğimiz (normalize etmediğimiz) için doğru sonuç verecektir.

# 1. BUN/CREA Oranı (Böbrek fonksiyonu göstergesi)
df['bun_crea_ratio'] = df['bun'] / df['crea']

# 2. PaFi (Oksijenizasyon İndeksi)
# PaFi değeri zaten mevcut, ancak bazı verilerde oksijen basıncı (paO2) ve solunum hızı (resp) gibi etkileşimler faydalı olabilir.
# Şimdilik en yaygın olan BUN/CREA oranını kullanalım.

# Oran hesaplama sırasında paydanın 0 olması durumunda oluşan Sonsuz (inf) değerleri yönetme
# Bunları o sütunun medyan değeri ile değiştiriyoruz
median_ratio = df['bun_crea_ratio'].median()
df['bun_crea_ratio'].replace([np.inf, -np.inf], median_ratio, inplace=True)

print("\nYeni Oran Sütunu ilk 5 değeri:")
print(df['bun_crea_ratio'].head())

# Eksik değerleri olan sütunları bulma (son bir kontrol)
cols_with_na = df.columns[df.isnull().any()].tolist()

# 'dnr' sütunundaki sayısal olmayan değerleri NaN yap
# Bu, medyan hesaplamadan önce sütunun tamamen sayısal olmasını sağlar.
df['dnr'] = pd.to_numeric(df['dnr'], errors='coerce')

# Eksik olan her sütunu kendi medyan değeri ile doldurun
for col in cols_with_na:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Toplam eksik değer sayısını kontrol edin (Sıfır olmalı!)
total_missing = df.isnull().sum().sum()
print(f"\nDoldurma sonrası toplam eksik değer: {total_missing}")

# Aykırı değer yönetimi yapılacak temel sayısal sütunlar (Örnekleme)
outlier_cols = ['wblc', 'meanbp', 'hrt', 'resp', 'temp', 'glucose', 'bun', 'crea']

for col in outlier_cols:
    # Alt sınır (%1'lik dilim) ve Üst sınır (%99'luk dilim) hesaplama
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)

    # Değerleri bu sınırlar içine hapsetme (Capping)
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("\nAykırı değer yönetimi tamamlandı.")

from sklearn.preprocessing import StandardScaler

# Ölçeklendirme (Standardization) yapılacak sütunları belirleme
# Hedef değişkenler ve ikili değişkenler (0/1) hariç tüm sayısal sütunlar
exclude_cols = ['Time', 'Event', 'sex', 'dnr', 'hospdead']
numerical_cols_to_scale = [col for col in df.columns if col not in exclude_cols]

scaler = StandardScaler()

# Sadece belirlenen sayısal sütunlara ölçeklendirme uygulayın
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

print("\nÖlçeklendirme sonrası veri setinin ilk 5 satırı (sayısal değerler 0 civarında olmalı):")
print(df.head())

# Hedef (Y): Sağkalım Analizi için (Time ve Event)
# Özellikler (X): Tahmin için kullanılacak tüm sütunlar

X = df.drop(columns=['Time', 'Event'])
# Survival Analysis modelleri genellikle Time ve Event sütunlarını Y olarak bir arada bekler.
y = df[['Time', 'Event']]

print("\nVeri Hazırlığı Tamamlandı! Model Kurulumuna Hazırız.")
print(f"Özellikler (X) Boyutu: {X.shape}")
print(f"Hedefler (Y) Boyutu: {y.shape}")

df.head(3)

# 'index=False' parametresini kullanmak önemlidir.
# Bu, pandas'ın otomatik olarak oluşturduğu fazladan satır numarası sütununu kaydetmemizi engeller.
output_file_name = 'survival_analysis_cleaned_data.csv'
df.to_csv(output_file_name, index=False)

print(f"Veri seti başarıyla '{output_file_name}' olarak kaydedildi.")

# Google Colab kütüphanesini kullanarak dosyayı indirin
from google.colab import files

files.download(output_file_name)

print("\nDosya indirme işlemi başlatıldı. Lütfen bilgisayarınızdaki indirme klasörünü kontrol edin.")

