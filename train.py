import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS


# ===============================
# 1. Load and preprocess data
# ===============================
df = pd.read_csv("data/trends.csv", skiprows=10)
df = df.drop(columns=["Rank", "Weekly change", "Monthly change", "Yearly change"])

# Function to create sequences
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# ===============================
# 2. Define LSTM model
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

# ===============================
# 3. Train model on a default trend
# ===============================
trend_name_default = "last minute halloween costumes"
row = df[df["Trend"] == trend_name_default].iloc[0]
ts = row.drop("Trend").astype(float)
df_ts = pd.DataFrame({"date": pd.to_datetime(ts.index), "score": ts.values})

scaler = MinMaxScaler()
df_ts["score_scaled"] = scaler.fit_transform(df_ts[["score"]])
scores = df_ts["score_scaled"].values

X, y = create_sequences(scores, window_size=10)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

train_loader = DataLoader(list(zip(X_train_t, y_train_t)), batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
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

# ===============================
# 4. Prediction function
# ===============================
def predict_next_week_score(trend_name, df, model, window_size=10):
    row = df[df["Trend"].str.lower().str.contains(trend_name.lower())]

    if row.empty:
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
# 5. Flask API
# ===============================
app = Flask(__name__)
CORS(app) 

@app.route("/predict")
def predict():
    trend = request.args.get("trend")
    if not trend:
        return jsonify({"error": "No trend provided"}), 400

    score = predict_next_week_score(trend, df, model)
    if score is None:
        return jsonify({"error": f"Trend '{trend}' not found"}), 404

    return jsonify({"predicted_score": float(score)})

@app.route("/top_trends")
def top_trends():
    trends_with_scores = []
    for _, row in df.iterrows():
        trend_name = row["Trend"]
        score = predict_next_week_score(trend_name, df, model)
        if score is not None:
            trends_with_scores.append({
                "trend": trend_name,
                "predicted_score": round(score)
            })
    trends_with_scores.sort(key=lambda x: x["predicted_score"], reverse=True)
    return jsonify(trends_with_scores[:7])



# Run Flask in a thread if this script is imported
def run_app():
    app.run(debug=True)

if __name__ == "__main__":
    run_app()
