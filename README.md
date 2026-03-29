# 🎮 India Gaming Ecosystem Intelligence Platform

An end-to-end machine learning platform that analyzes India's gaming ecosystem across **312 curated game records** — delivering game recommendations, sentiment analysis, market insights, and an interactive Streamlit dashboard.

> **Note on scale:** The platform processes **97,000+ similarity matrix computations** via the Cosine Similarity engine (312 × 312 game-to-game comparisons), making it computationally meaningful despite the focused, curated dataset.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 📌 What It Does

| Feature | Description |
|---|---|
| 🤖 ML Classification | Random Forest classifier predicts game category — **84% accuracy, 5-fold cross-validation** |
| 🎯 Recommendations | Cosine Similarity engine computes 97,000+ pairwise comparisons to suggest similar games |
| 💬 Sentiment Analysis | VADER NLP analyzes user reviews and classifies sentiment as Positive / Neutral / Negative |
| 📊 Market Analytics | SQL-powered queries on SQLite surface market trends by genre, platform, and region |
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
    └── games_dataset.csv   # 312-record curated Indian gaming dataset
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

| Metric | Result |
|---|---|
| Classification accuracy | **84%** (Random Forest, 5-fold CV) |
| Similarity computations | **97,000+** (full 312×312 cosine matrix) |
| Dataset size | **312** curated Indian gaming records |
| Recommendation engine | Top-N similar games via Cosine Similarity |
| Sentiment engine | VADER — Positive / Neutral / Negative |
| Analytics layer | SQL queries on SQLite via Pandas |

---

## 💡 Why 97,000+ Computations?

The Cosine Similarity recommendation engine computes a **full 312 × 312 similarity matrix** — every game compared against every other game in both directions. That equals **97,344 individual similarity computations**, making the recommendation engine responsive across the entire dataset without pre-filtering.

---

## 👤 Developer

**Kartikeya Babaraoji Warhade**
[LinkedIn](https://linkedin.com/in/kartikeya-warhade) · [GitHub](https://github.com/KartikeyaWarhade2002)
