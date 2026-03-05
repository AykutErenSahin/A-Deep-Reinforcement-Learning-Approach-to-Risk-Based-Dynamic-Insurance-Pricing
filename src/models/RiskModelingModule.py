
# =============================================================================
# PROJE ADI: Survival Analysis & DeepSurv Risk Scoring
# AÇIKLAMA: Bu kod, hasta verilerini kullanarak sağ kalım analizi yapar,
#           Deep Learning (DeepSurv) modeli ile risk skorlarını hesaplar
#           ve sonuçları fiyatlandırma (RL) aşaması için dışa aktarır.
# =============================================================================

# 1. KÜTÜPHANELER VE KURULUM
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib
# Tekrarlanabilirlik için tohum (seed) sabitleme
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME (PREPROCESSING)
# =============================================================================
print("--- [ADIM 1] Veri Yükleniyor ve Hazırlanıyor ---")

# Veri setini yükle
filename = 'CleanedClinicalData.csv'
df = pd.read_csv(filename)

# Hedef değişkenler ve Özelliklerin ayrılması
# Time: Geçen süre, Event: Olayın gerçekleşip gerçekleşmediği (1: Öldü, 0: Sansürlü)
target_cols = ['Time', 'Event']
feature_cols = [c for c in df.columns if c not in target_cols]

X = df[feature_cols].values
y_time = df['Time'].values
y_event = df['Event'].values

# Eğitim ve Test seti ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

# Standardizasyon (Deep Learning için KRİTİK)
# Veriler ortalama 0, standart sapma 1 olacak şekilde ölçeklenir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final çıktı için tüm veriyi de ölçekliyoruz
X_all_scaled = scaler.transform(X)

# Verileri PyTorch Tensor formatına çevirme (GPU/CPU işleyebilsin diye)
X_train_t = torch.FloatTensor(X_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
X_all_t = torch.FloatTensor(X_all_scaled)
y_time_train_t = torch.FloatTensor(y_time_train)
y_event_train_t = torch.FloatTensor(y_event_train)

print(f"Veri Hazır. Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

# =============================================================================
# 3. DEEPSURV MODEL MİMARİSİ (NEURAL NETWORK)
# =============================================================================
class DeepSurv(nn.Module):
    def __init__(self, in_features):
        super(DeepSurv, self).__init__()
        # 3 Katmanlı Sinir Ağı Yapısı
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),       # Giriş Katmanı
            nn.ReLU(),                        # Aktivasyon Fonksiyonu
            nn.Dropout(0.2),                  # Overfitting önleyici
            nn.Linear(64, 32),                # Gizli Katman
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)                  # ÇIKIŞ: Tek bir "Log-Risk" skoru
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# 4. KAYIP FONKSİYONU (COX PARTIAL LIKELIHOOD LOSS)
# =============================================================================
def deepsurv_loss(risk_pred, time_true, event_true):
    """
    Cox Proportional Hazards modelinin negatif log-likelihood fonksiyonu.
    Modelin, erken ölen hastalara daha yüksek risk puanı vermesini sağlar.
    """
    # Hastaları süreye göre sırala (Büyükten küçüğe)
    mask = torch.argsort(time_true, descending=True)
    risk_pred_sorted = risk_pred[mask]
    event_true_sorted = event_true[mask]

    # Formül: Risk = exp(h(x))
    exp_risk = torch.exp(risk_pred_sorted)

    # Risk Seti Kümülatif Toplamı
    risk_set_sum = torch.cumsum(exp_risk, dim=0)

    # Log-Likelihood hesaplama
    log_risk_set_sum = torch.log(risk_set_sum)
    diff = risk_pred_sorted - log_risk_set_sum

    # Sadece olayın gerçekleştiği (Event=1) durumlarda hatayı hesapla
    num_events = torch.sum(event_true_sorted)
    if num_events == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = -torch.sum(diff * event_true_sorted.view(-1, 1)) / num_events
    return loss

# =============================================================================
# 5. MODEL EĞİTİMİ (TRAINING LOOP)
# =============================================================================
print("\n--- [ADIM 2] Model Eğitimi Başlıyor ---")

model = DeepSurv(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.005) # Learning Rate

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # İleri Yayılım
    pred = model(X_train_t)

    # Kayıp Hesapla
    loss = deepsurv_loss(pred, y_time_train_t, y_event_train_t)

    # Geri Yayılım
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

# Eğitim Kaybı Grafiği
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('DeepSurv Model Eğitim Süreci')
plt.xlabel('Epochs')
plt.ylabel('Cox Loss')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 6. MODEL DEĞERLENDİRME (EVALUATION - C-INDEX)
# =============================================================================
print("\n--- [ADIM 3] Model Performans Değerlendirmesi ---")

def calculate_c_index(risk_scores, times, events):
    """
    Concordance Index (C-Index): Modelin sıralama başarısı.
    1.0 = Mükemmel, 0.5 = Rastgele.
    """
    n = len(times)
    # Vektörize edilmiş hesaplama (Hızlı)
    T_i, T_j = np.meshgrid(times, times)
    R_i, R_j = np.meshgrid(risk_scores, risk_scores)
    E_i, _ = np.meshgrid(events, events)

    # Karşılaştırılabilir: T_i < T_j ve i kişisi ölmüşse
    admissible = (T_i < T_j) & (E_i == 1)

    # Uyumlu: Erken ölenin (i) riski daha yüksek olmalı (R_i > R_j)
    concordant = (R_i > R_j) & admissible

    if np.sum(admissible) == 0: return 0.5
    return np.sum(concordant) / np.sum(admissible)

model.eval()
with torch.no_grad():
    test_pred = model(X_test_t).numpy().flatten()
    c_index_score = calculate_c_index(test_pred, y_time_test, y_event_test)

print(f"Test Seti C-Index Skoru: {c_index_score:.4f}")
print("Yorum: 0.7 üzeri iyi, 0.8 üzeri çok iyi kabul edilir.")

# =============================================================================
# 7. ÇIKTI ÜRETME (LOG-RISK & LAMBDA)
# =============================================================================
print("\n--- [ADIM 4] Log-Risk Skorları ve Dosya Kaydı ---")

with torch.no_grad():
    # Tüm veri seti için Log-Risk skorlarını hesapla
    all_log_risk = model(X_all_t).numpy().flatten()

# DataFrame'e yeni sütunları ekle
df['log_risk_score'] = all_log_risk

# Lambda (Hazard Multiplier) Hesabı
# λ(x) = exp(h(x)) -> Kişinin temel riske göre kaç kat riskli olduğu
df['hazard_multiplier'] = np.exp(all_log_risk)

# Sonuçları kaydet
output_filename = 'survival_analysis_final_with_risk_scores.csv'
df.to_csv(output_filename, index=False)

print(f"\nİŞLEM BAŞARIYLA TAMAMLANDI.")
print(f"1. Oluşturulan dosya: {output_filename}")
print(f"2. Dosyadaki yeni sütunlar: 'log_risk_score' ve 'hazard_multiplier'")
print("Bu dosya artık RL (Fiyatlandırma) algoritmasına girmeye hazırdır.")

torch.save(model.state_dict(), 'risk_model.pth')
print("Risk Modeli (risk_model.pth) kaydedildi.")
joblib.dump(scaler, 'scaler_risk.pkl')
