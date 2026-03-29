
# 🎮 India Gaming Ecosystem Intelligence Platform

An end-to-end machine learning platform that analyzes India's gaming ecosystem across **312 curated game records** — delivering game recommendations, sentiment analysis, market insights, and an interactive Streamlit dashboard.

> **Note on scale:** The Cosine Similarity engine computes a full **312 × 312 similarity matrix — 97,000+ pairwise computations** — comparing every game against every other game to power recommendations.

---

## 🚀 Live Demo

> 🌐 **[Live App → gaming-analytics-kartikeya.streamlit.app](https://gaming-analytics-kartikeya.streamlit.app/)**

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
| Similarity computations | **97,000+** (full 312 × 312 cosine matrix) |
| Dataset | **312** curated Indian gaming records |
| Recommendation engine | Top-N similar games via Cosine Similarity |
| Sentiment engine | VADER — Positive / Neutral / Negative |
| Analytics layer | SQL queries on SQLite |

---

## 💡 Why 97,000+ Computations?

The recommendation engine builds a **full 312 × 312 similarity matrix** — every game compared against every other in both directions. That equals **97,344 individual similarity computations**, making the engine responsive across the entire dataset with no pre-filtering needed.

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
