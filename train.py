import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. Load and preprocess data
# ===============================
df = pd.read_csv("data/trends.csv", skiprows=10)
df = df.drop(columns=["Rank", "Weekly change", "Monthly change", "Yearly change"])

# Choose a trend to train on (we'll allow user to choose later)
trend_name_default = "last minute halloween costumes"
row = df[df["Trend"] == trend_name_default].iloc[0]

# Convert wide row → time-series
ts = row.drop("Trend").astype(float)
df_ts = pd.DataFrame({"date": pd.to_datetime(ts.index), "score": ts.values})

# Normalize
scaler = MinMaxScaler()
df_ts["score_scaled"] = scaler.fit_transform(df_ts[["score"]])
scores = df_ts["score_scaled"].values

# Create sequences
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(scores, window_size=10)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Dataset & loader
class TrendDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TrendDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TrendDataset(X_test_t, y_test_t), batch_size=32)

# ===============================
# 2. Define the LSTM model
# ===============================
class TrendLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.fc(h[-1])
        return out

model = TrendLSTM(hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 3. Train the model
# ===============================
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Test MSE
model.eval()
with torch.no_grad():
    preds = model(X_test_t)
    test_loss = criterion(preds, y_test_t)
    print("Test MSE:", test_loss.item())

# ===============================
# 4. Function to predict next week
# ===============================
def predict_next_week_score(trend_name, df, model, window_size=10):
    row = df[df["Trend"] == trend_name]
    if row.empty:
        print(f"Trend '{trend_name}' not found.")
        return None
    row = row.iloc[0]
    ts = row.drop("Trend").astype(float)
    ts_min, ts_max = ts.min(), ts.max()
    scores = (ts - ts_min) / (ts_max - ts_min)
    last_window = scores[-window_size:]
    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred_scaled = model(x).item()
    pred = pred_scaled * (ts_max - ts_min) + ts_min
    return pred

# ===============================
# 5. Prompt user for trend
# ===============================
trend_name_input = input("Enter a Pinterest trend name: ")
score = predict_next_week_score(trend_name_input, df, model)
if score is not None:
    print(f"Predicted next week's score for '{trend_name_input}': {score:.2f}")

# Optional: save model
torch.save(model.state_dict(), "trend_lstm.pth")
