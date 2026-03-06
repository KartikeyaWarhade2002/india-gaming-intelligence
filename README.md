# 🎮 India Gaming Ecosystem Intelligence Platform

An end-to-end machine learning platform that analyzes India's gaming ecosystem across **20,000+ records** — delivering game recommendations, sentiment analysis, market insights, and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 📌 What It Does

| Feature | Description |
|---|---|
| 🤖 ML Classification | Random Forest classifier predicts game category (84% accuracy, 5-fold CV) |
| 🎯 Recommendations | Cosine Similarity engine suggests similar games based on features |
| 💬 Sentiment Analysis | VADER NLP analyzes user reviews and classifies sentiment |
| 📊 Market Analytics | SQL-powered queries on SQLite for market trends and ecosystem insights |
| 🖥️ Dashboard | Interactive Streamlit app with real-time charts and filters |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading, cleaning, manipulation |
| Scikit-learn | Random Forest classifier, Cosine Similarity, cross-validation |
| NLTK / VADER | NLP sentiment analysis |
| SQLite | Market analytics queries |
| Streamlit | Interactive web dashboard |
| Matplotlib / Seaborn | Visualizations |

---

## 📁 Project Structure

```
india-gaming-intelligence/
│
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # All dependencies
├── README.md
│
└── data/
    └── games_dataset.csv   # Dataset (20,000+ records)
```

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/KartikeyaWarhade2002/india-gaming-intelligence.git
cd india-gaming-intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Save as `requirements.txt` in the project folder.

---

## 🔍 Key Results

- **84% accuracy** on game category classification (Random Forest, 5-fold CV)
- **20,000+ records** processed across the full ML pipeline
- Cosine Similarity recommendation engine returns top-N similar games
- VADER sentiment classifies reviews as Positive / Neutral / Negative
- SQL analytics surface market trends by genre, platform, and region

---

## 📸 Screenshots

**Dashboard Overview**
<img width="7684" height="4322" alt="dashboard" src="https://github.com/user-attachments/assets/0572a433-8bd5-4cf3-8740-3e24ffbf1545" />


**Game Recommender**
<img width="7684" height="4322" alt="recommendations" src="https://github.com/user-attachments/assets/bd9247dd-c253-4f81-8137-08c097419a79" />



**Sentiment Analysis**
<img width="7684" height="4322" alt="sentiment" src="https://github.com/user-attachments/assets/aff7da20-f4f7-48a7-883e-2e4b31937ee1" />



**Trending Games**
<img width="7684" height="4322" alt="analytics" src="https://github.com/user-attachments/assets/86d306af-d4a4-4cd2-adc0-467d2a2334ee" />


---

## 👤 Developer

**Kartikeya Warhade**
[LinkedIn](https://linkedin.com/in/kartikeya-warhade) · [GitHub](https://github.com/KartikeyaWarhade2002)
