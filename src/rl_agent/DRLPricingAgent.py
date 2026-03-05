import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Start noktası şekli
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

"""**Veri Yükleme**"""

df = pd.read_csv('ImputedTrainingData.csv')

exclude_cols = ['Event', 'Time', 'death_age', 'log_risk_score', 'is_imputed', 'slos', 'hospdead']

state_columns = [col for col in df.columns if col not in exclude_cols]

"""**Environment**"""

class InsuranceEnv(gym.Env):
    def __init__(self, df, state_cols):
        super(InsuranceEnv, self).__init__()

        self.df = df
        self.state_cols = state_cols
        self.current_idx = 0

        self.reputation = 1.0
        self.rep_decay = 0.02
        self.rep_boost = 0.005

        self.price_multipliers = [0.8, 1.0, 1.2, 1.5]
        self.action_space = gym.spaces.Discrete(len(self.price_multipliers))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(state_cols)+ 1,), dtype=np.float32
        )

    def reset(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.current_idx = 0
        self.reputation = 1.0
        state = self.df.iloc[self.current_idx][self.state_cols].values
        state = np.append(state, self.reputation)
        return state.astype(np.float32)

    def step(self, action):
        customer = self.df.iloc[self.current_idx]

        risk_factor = customer['hazard_multiplier']
        cost = customer['charges']

        base_premium = cost * 1.1

        chosen_multiplier = self.price_multipliers[action]
        offered_price = base_premium * chosen_multiplier

        price_ratio = offered_price / cost

        prob_buy = np.exp(-2.0 * (price_ratio - 0.8))
        prob_buy = np.clip(prob_buy, 0, 1)

        final_prob_buy = prob_buy * (0.2 + 0.8 * self.reputation)

        is_sold = np.random.random() < final_prob_buy

        reward = 0
        actual_claim_cost = 0

        if is_sold:
            base_accident_prob = 0.15
            self.reputation = min(1.0, self.reputation + self.rep_boost)
            customer_prob = min(base_accident_prob * risk_factor, 1.0)

            accident_happened = np.random.random() < customer_prob

            if accident_happened:
                actual_claim_cost = np.random.gamma(shape=5.0, scale=cost/5.0)
                reward = offered_price - actual_claim_cost
            else:
                reward = offered_price

        else:
            excess_greed = max(0, price_ratio - 1.0)
            penalty = excess_greed * 0.05

            self.reputation = max(0.0, self.reputation - penalty)

            reward = 0

        # Sonraki müşteri
        self.current_idx += 1
        done = self.current_idx >= len(self.df) - 1

        if not done:
            next_customer_data = self.df.iloc[self.current_idx][self.state_cols].values
            next_state = np.append(next_customer_data, self.reputation)
        else:
            next_state = np.zeros(len(self.state_cols) + 1)

        info = {
            "is_sold": is_sold,
            "price": offered_price,
            "claim": actual_claim_cost,
            "cost": cost,
            "reputation": self.reputation
        }

        return next_state.astype(np.float32), reward, done, info

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Initialize target model
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def update_target_model(self):
        """Copies weights from primary model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Tensor dönüşümleri (Toplu işlem hızı için)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-Learning Formülü: Q_new = Reward + Gamma * max(Q_next)
        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item() if loss is not None else None

import warnings
warnings.filterwarnings('ignore')

env = InsuranceEnv(df, state_columns)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

EPISODES = 20
rewards_history = []
loss_history = []

TARGET_UPDATE_FREQ = 10 # Define the target update frequency

history = {
    "losses": [],
    "avg_q_values": [],
    "rewards": [],
    "actions": [],
    "profits": [],
    "reputations": []
}

for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(agent.device)
            q_values = agent.model(state_tensor)
            max_q = q_values.max().item()
            history["avg_q_values"].append(max_q)

        action = agent.act(state)
        history["actions"].append(action)

        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay() # replay returns loss, capture it here
        if loss is not None:
            history["losses"].append(loss)

        state = next_state
        total_reward += reward
        step_count += 1

    history["rewards"].append(total_reward)
    history["profits"].append(total_reward)
    history["reputations"].append(env.reputation)

    rewards_history.append(total_reward)
    print(f"Episode: {e+1}/{EPISODES} | Toplam Kâr: ${total_reward:,.0f} | Epsilon: {agent.epsilon:.2f}")

    # Update the target network periodically
    if (e + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()
        print("Target Network Güncellendi")

print("Eğitim Tamamlandı.")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Görsel stil ayarı
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('ggplot')

def plot_academic_results(history, window_size=50):
    """
    history: Verilerin olduğu sözlük (reputations dahil)
    window_size: Grafikleri yumuşatmak için hareketli ortalama penceresi
    """

    # Yardımcı Fonksiyon: Moving Average
    def moving_average(data, window):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Figür boyutu (A4 dikey)
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))

    # BAŞLIK AYARI (Çakışmayı önlemek için y=0.96 yaptık)
    fig.suptitle('DRL Ajanı Eğitim Analizi (Reputation & Profit)', fontsize=18, fontweight='bold', y=0.96)

    # ---------------------------------------------------------
    # 1. LOSS GRAFİĞİ
    # ---------------------------------------------------------
    ax = axs[0, 0]
    losses = history.get("losses", [])
    if len(losses) > 0:
        ax.plot(losses, alpha=0.3, color='gray', label='Raw Loss')
        smooth_loss = moving_average(losses, window_size)
        ax.plot(smooth_loss, color='red', linewidth=2, label=f'Trend ({window_size})')
        ax.set_title('Training Loss (Hata)', fontweight='bold')
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 2. AVERAGE Q-VALUE
    # ---------------------------------------------------------
    ax = axs[0, 1]
    qs = history.get("avg_q_values", [])
    if len(qs) > 0:
        ax.plot(qs, color='purple', alpha=0.8)
        ax.set_title('Average Max Q-Value (Beklenen Ödül)', fontweight='bold')
        ax.set_ylabel('Q Value')
        ax.set_xlabel('Steps')
        ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 3. EPISODE REWARDS
    # ---------------------------------------------------------
    ax = axs[1, 0]
    rewards = history.get("rewards", [])
    if len(rewards) > 0:
        ax.plot(rewards, alpha=0.3, color='orange')
        smooth_rw = moving_average(rewards, window_size)
        ax.plot(smooth_rw, color='brown', linewidth=2, label='Trend')
        ax.set_title('Total Reward per Episode', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 4. KÜMÜLATİF KAR (PROFIT)
    # ---------------------------------------------------------
    ax = axs[1, 1]
    profits = history.get("profits", [])
    if len(profits) > 0:
        cumulative_profit = np.cumsum(profits)
        ax.plot(cumulative_profit, color='green', linewidth=2)
        ax.fill_between(range(len(cumulative_profit)), cumulative_profit, color='green', alpha=0.1)
        ax.set_title('Cumulative Profit (Toplam Kasa)', fontweight='bold')
        ax.set_ylabel('Para Birimi')
        ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 5. ACTION DISTRIBUTION (Histogram)
    # ---------------------------------------------------------
    ax = axs[2, 0]
    actions = history.get("actions", [])
    if len(actions) > 100:
        n = len(actions)
        part_size = int(n * 0.1)
        parts = [
            actions[:part_size],                # Başlangıç
            actions[int(n*0.45):int(n*0.55)],   # Orta
            actions[-part_size:]                # Son
        ]
        labels = ['Başlangıç', 'Orta', 'Son']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # Dinamik bin sayısı (Action sayısını otomatik bulur)
        max_act = max(actions) if len(actions) > 0 else 3
        bins = np.arange(max_act + 2) - 0.5

        ax.hist(parts, label=labels, color=colors, bins=bins, rwidth=0.8)
        ax.set_title('Action Distribution (Strateji Değişimi)', fontweight='bold')
        ax.set_xlabel('Action ID')
        ax.set_ylabel('Frekans')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Yeterli veri yok", ha='center')

    # ---------------------------------------------------------
    # 6. REPUTATION CHANGE (YENİ EKLENEN - SAĞ ALT)
    # ---------------------------------------------------------
    ax = axs[2, 1]
    reps = history.get("reputations", [])
    if len(reps) > 0:
        ax.plot(reps, color='teal', linewidth=2)
        ax.set_ylim(0, 1.1) # 0 ile 1 arasında sabitler
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Max')

        ax.set_title('Average Reputation per Episode', fontweight='bold')
        ax.set_ylabel('Reputation Score (0-1)')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        # Yorum: Eğer DRL çalışıyorsa bu çizgi 0'a çakılmamalı, dengede durmalı.
    else:
        ax.text(0.5, 0.5, "Reputation verisi yok", ha='center')

    # LAYOUT DÜZELTME (Başlık için üstten boşluk bırakıyoruz)
    # rect=[sol, alt, sağ, üst] -> üst sınırı 0.94 yaptık ki başlık sığsın
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    plt.savefig('akademik_rapor_grafikleri.png', dpi=300)
    plt.show()

# Kullanımı:
plot_academic_results(history)

torch.save(agent.model.state_dict(), 'dqn_policy.pth')
print("DQN Policy Modeli (dqn_policy.pth) başarıyla kaydedildi.")
