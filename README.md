# 📌 Pinterest Trend Predictor
This project predicts the next week's popularity score of Pinterest trends using an **LSTM (Long Short-Term Memory)** model. It reads historical trend data from a CSV file exported from Pinterest and provides predictions through a **Flask API**.

---

## ✨ Features

* 📊 Load and preprocess Pinterest trends CSV data.
* 🧠 Train an LSTM model on a selected Pinterest trend.
* 🔮 Predict next week’s score for any trend.
* 🌐 Flask API with endpoints:

  * `/predict` → Predict score for a specific Pinterest trend.
  * `/top_trends` → Get the top predicted Pinterest trends.
* 🌍 CORS enabled to allow frontend requests from any origin.

---

## 🛠 Requirements

* Python 3.8+
* Packages:

  ```bash
  pandas
  numpy
  torch
  scikit-learn
  flask
  flask-cors
  ```

Install dependencies:

```bash
pip install pandas numpy torch scikit-learn flask flask-cors
```

---

## 📁 Project Structure

```
.
├── data/
│   └── trends.csv          # Pinterest CSV file with trend data
├── index.html
├── train.py                # Main script: training + Flask API
└── README.md
```

---

## ⚙️ How It Works

1. **Load Data** 📂

   * Reads `trends.csv` exported from Pinterest.
   * Drops irrelevant columns and keeps trend names and historical scores.

2. **Create Sequences** 🔢

   * Converts historical trend scores into sequences for LSTM input.
   * Each sequence of `window_size` scores predicts the next score.

3. **LSTM Model** 🧠

   * LSTM processes sequences to learn patterns in trend popularity.
   * Output layer predicts the next week’s trend score.

4. **Train Model** 🏋️

   * Trains on a default trend using Mean Squared Error (MSE) loss.
   * Uses the Adam optimizer for weight updates.

5. **Prediction** 🔮

   * Function `predict_next_week_score(trend_name, df, model)` predicts next week’s score for any Pinterest trend.

6. **Flask API** 🌐

   * `/predict?trend=<trend_name>` → Returns JSON with predicted score.
   * `/top_trends` → Returns top 7 trends with highest predicted scores.

---

## 🚀 Usage

1. **Run the Backend**

```bash
python train.py
```

2. **Run the Frontend**

```bash
Open the index.html file in the local file projects
```

3. **Predict a trend**

   * Example GET request:

```
http://127.0.0.1:5000/predict?trend=halloween decorations
```

4. **Get top trends**

   * Example GET request:

```
http://127.0.0.1:5000/top_trends
```

---

## 💡 Notes

* Make sure your CSV file includes **trend names** and **historical weekly scores**.
* The model is trained on a default trend; for better predictions, retrain on multiple trends.
* Predictions are scaled back to the original Pinterest score range for accuracy.

---

📌 Made with ❤️ for Pinterest trend enthusiasts and data lovers.
