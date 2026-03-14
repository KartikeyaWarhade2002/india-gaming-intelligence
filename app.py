# =============================================================================
# PROJECT 1: India Gaming Ecosystem Intelligence Platform
# Developer: Kartikeya Warhade
#
# CHANGES IN THIS VERSION:
#   1. Recommendation slider: dynamic max = number of similar games found
#   2. Removed "How Cosine Similarity Works" expander from UI
#   3. Fixed duplicate: BGMI and Battlegrounds Mobile India merged to BGMI
#   4. Sentiment result colours changed to dark background for readability
#   5. ML Predictor year range extended to 2030
#   5b. Custom CSV upload added — feed your own games dataset
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import sqlite3
import io
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(csv_path=None):
    """
    Load games dataset. Accepts either:
    - Default path (games_clean.csv next to app.py)
    - A custom uploaded file bytes object
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        default = os.path.join(os.path.dirname(__file__), "games_clean.csv")
        if not os.path.exists(default):
            st.error("games_clean.csv not found. Place it in the same folder as app.py")
            st.stop()
        df = pd.read_csv(default)

    df.columns = [c.strip() for c in df.columns]

    numeric_cols = ['multiplayer','battle_royale','open_world','fps_shooter',
                    'sports','rpg','indie','review_score',
                    'player_count_millions','indian_popularity','release_year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


@st.cache_resource
def build_database(df):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("games", conn, if_exists="replace", index=False)
    return conn


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

FEATURE_COLS = ['multiplayer','battle_royale','open_world','fps_shooter',
                'sports','rpg','indie','review_score',
                'player_count_millions','indian_popularity','release_year']


@st.cache_data
def build_similarity_matrix(df):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[FEATURE_COLS])
    sim_matrix = cosine_similarity(features)
    return sim_matrix


def get_all_similar_games(df, sim_matrix, game_name, platform_filter="All", threshold=0.80):
    """
    FIX 1: Returns ALL games above a similarity threshold.
    The caller then uses the count to set the slider max dynamically.

    threshold=0.80 means: only show games that are at least 80% similar.
    This ensures the slider max is meaningful — not padded with unrelated games.
    """
    matches = df[df['name'] == game_name]
    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches.index[0]
    sim_scores = sim_matrix[idx].copy()

    results = df.copy()
    results['similarity_score'] = sim_scores
    results = results[results['name'] != game_name]

    if platform_filter != "All":
        results = results[results['platform'] == platform_filter]

    # Only games above threshold
    results = results[results['similarity_score'] >= threshold]
    results = results.sort_values('similarity_score', ascending=False)

    return results[['name','genre','platform','review_score',
                     'indian_popularity','similarity_score']].round(3)


# =============================================================================
# RANDOM FOREST
# =============================================================================

@st.cache_data
def train_random_forest(df):
    df_model = df.copy()
    df_model['target'] = (df_model['indian_popularity'] >= 8).astype(int)

    # Only use rows where all feature columns exist
    df_model = df_model.dropna(subset=FEATURE_COLS)

    X = df_model[FEATURE_COLS]
    y = df_model['target']

    if len(y.unique()) < 2:
        return None, 0, np.array([0]), pd.Series(), None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        max_depth=6, min_samples_split=5
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

    importances = pd.Series(
        rf.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return rf, accuracy, cv_scores, importances, X_test, y_test, y_pred


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

SAMPLE_REVIEWS = {
    "BGMI": [
        "BGMI is absolutely amazing, best battle royale on mobile!",
        "Great game but too many hackers ruining the experience.",
        "BGMI is the best game ever made for Indian gamers!!",
        "Servers crash too often, very frustrating gameplay.",
        "Love the graphics and smooth controls on my phone.",
        "Krafton really improved the anti-cheat system recently.",
        "Not bad but Free Fire has better content updates.",
        "BGMI tournaments have great prize pools, very competitive.",
        "The new map is incredible, gameplay feels fresh again.",
        "Too much pay to win. Disappointed with recent updates."
    ],
    "Free Fire": [
        "Free Fire is the king of mobile battle royale in India!",
        "Smooth gameplay even on low-end devices, great optimization.",
        "The characters and abilities make it unique and fun.",
        "Too many bots in low rank games, not challenging enough.",
        "Garena keeps releasing amazing content every month.",
        "Graphics could be better but gameplay is very addictive.",
        "OB events are always exciting, love the limited items.",
        "Server lag is a big problem during peak hours.",
        "Best game for budget phones, runs on anything.",
        "Free Fire MAX is even better with the enhanced graphics."
    ],
    "Valorant": [
        "Best tactical FPS I have ever played, amazing gunplay.",
        "The agents and abilities add a great strategic layer.",
        "Riot Games keeps the game fresh with constant updates.",
        "Ranked mode is frustrating due to smurfs and cheaters.",
        "Graphics are clean and performance is excellent even on old PCs.",
        "The community is toxic at lower ranks unfortunately.",
        "Amazing competitive scene and esports tournaments.",
        "Perfect game for Indian esports aspirants.",
        "The new map design keeps matches interesting and varied.",
        "Viper and Jett are way too overpowered right now."
    ],
    "GTA V": [
        "GTA V is an absolute masterpiece of open world design.",
        "GTA Online has so much content you never get bored.",
        "Rockstar keeps milking old game instead of releasing GTA VI.",
        "The story mode is one of the best ever written in gaming.",
        "GTA Online loading times are terrible even in 2024.",
        "Amazing game but griefers ruin the online experience.",
        "Best open world game ever made, incredible attention to detail.",
        "Modding community keeps this game alive after all these years.",
        "Character switching mechanic is genuinely innovative.",
        "Graphics hold up incredibly well even after 10 years."
    ],
    "God of War PS4": [
        "God of War PS4 is the greatest PlayStation game ever made!",
        "Kratos and Atreus relationship is incredibly well written.",
        "The combat system is deep and satisfying from start to finish.",
        "Absolutely stunning visuals and one-shot camera is genius.",
        "The boss fights are epic in scale and challenge.",
        "Best story in any action game I have experienced.",
        "The Norse mythology setting is fresh and beautifully realized.",
        "Performance on PS4 base is a bit rough in some areas.",
        "Ragnarok improved on this in every way possible.",
        "This game single-handedly saved Sony's reputation."
    ]
}


def analyze_sentiment(reviews):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for review in reviews:
        scores = analyzer.polarity_scores(review)
        compound = scores['compound']
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({
            'Review': review,
            'Compound': round(compound, 3),
            'Sentiment': label
        })
    return pd.DataFrame(results)


# =============================================================================
# CHART HELPERS
# =============================================================================

def make_bar_chart(data, x, y, title, color='#1565C0', horizontal=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    if horizontal:
        ax.barh(data[x], data[y], color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel(y)
        ax.set_ylabel(x)
    else:
        ax.bar(data[x], data[y], color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=35, ha='right', fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def make_pie_chart(labels, values, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1565C0','#0288D1','#00838F','#2E7D32','#558B2F','#F9A825','#E65100']
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=colors[:len(labels)], startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="India Gaming Intelligence Platform",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }

    /* FIX 4: Dark background sentiment table rows */
    .sentiment-positive {
        background-color: #1B5E20 !important;
        color: white !important;
        padding: 6px 10px;
        border-radius: 4px;
        margin: 2px 0;
        font-size: 14px;
    }
    .sentiment-negative {
        background-color: #B71C1C !important;
        color: white !important;
        padding: 6px 10px;
        border-radius: 4px;
        margin: 2px 0;
        font-size: 14px;
    }
    .sentiment-neutral {
        background-color: #E65100 !important;
        color: white !important;
        padding: 6px 10px;
        border-radius: 4px;
        margin: 2px 0;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── SESSION STATE ─────────────────────────────────────────────────────────
    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dataset_label" not in st.session_state:
        st.session_state.dataset_label = ""
    # Track which source is ACTIVE so Load Default can override an uploaded file
    if "active_source" not in st.session_state:
        st.session_state.active_source = None   # "default" | "custom"

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎮 Gaming Intelligence")
        st.markdown("---")
        st.subheader("📂 Dataset")

        # ── BUTTON 1: Load Default Dataset ────────────────────────────────────
        if st.button("📦 Load Default Dataset", use_container_width=True,
                     help="Load the built-in 312-game dataset instantly"):
            with st.spinner("Loading default dataset..."):
                st.session_state.df           = load_data()
                st.session_state.dataset_loaded = True
                st.session_state.dataset_label  = "default"
                st.session_state.active_source  = "default"
            st.success(f"✅ Default dataset loaded: {len(st.session_state.df)} games")

        st.markdown(
            '<div style="text-align:center; color:#888; font-size:12px; '
            'margin:8px 0;">— or upload a new one below —</div>',
            unsafe_allow_html=True
        )

        # ── FILE UPLOADER (hidden label — button below does the loading) ───────
        uploaded_csv = st.file_uploader(
            "Upload CSV",
            type=['csv'],
            label_visibility="collapsed",   # hides the "Load New Dataset" text label
            help=(
                "Upload any CSV with columns:\n"
                "name, genre, platform, multiplayer, battle_royale,\n"
                "open_world, fps_shooter, sports, rpg, indie,\n"
                "review_score, player_count_millions, indian_popularity,\n"
                "release_year, developer, publisher"
            )
        )

        # ── BUTTON 2: Load New Dataset (only active when a file is staged) ─────
        btn_disabled = uploaded_csv is None
        if st.button(
            "📂 Load New Dataset",
            use_container_width=True,
            disabled=btn_disabled,
            help="Click after selecting a CSV file above"
        ):
            try:
                with st.spinner("Loading your dataset..."):
                    st.session_state.df           = load_data(uploaded_csv)
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_label  = "custom"
                    st.session_state.active_source  = "custom"
                st.success(f"✅ Custom dataset loaded: {len(st.session_state.df)} games")
                st.info("All 5 tabs now use your uploaded data.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

        # ── Status indicator ──────────────────────────────────────────────────
        if st.session_state.active_source == "default":
            st.markdown(
                '<div style="background:#1B5E20; color:white; border-radius:6px; '
                'padding:6px 10px; font-size:12px; margin-top:6px; text-align:center;">'
                f'✅ Default dataset active — {len(st.session_state.df) if st.session_state.df is not None else 0} games'
                '</div>',
                unsafe_allow_html=True
            )
        elif st.session_state.active_source == "custom":
            st.markdown(
                '<div style="background:#0D47A1; color:white; border-radius:6px; '
                'padding:6px 10px; font-size:12px; margin-top:6px; text-align:center;">'
                f'✅ Custom dataset active — {len(st.session_state.df) if st.session_state.df is not None else 0} games'
                '</div>',
                unsafe_allow_html=True
            )

        # ── Dataset coverage (only shown after loading) ────────────────────────
        if st.session_state.dataset_loaded and st.session_state.df is not None:
            st.markdown("---")
            st.markdown("**Dataset Coverage**")
            df_side = st.session_state.df
            for platform, count in df_side['platform'].value_counts().items():
                pct = count / len(df_side) * 100
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between;'
                    f'font-size:13px; padding:2px 0;">'
                    f'<span>• {platform}</span>'
                    f'<span style="color:#aaa;">{count} ({pct:.0f}%)</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("---")
            st.markdown(
                '<div style="color:#888; font-size:12px; text-align:center;">'
                'No dataset loaded yet.<br>Click a button above to begin.</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.caption("Built by Kartikeya Warhade")

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.title("🎮 India Gaming Ecosystem Intelligence Platform")
    st.markdown("**ML-powered · Built by Kartikeya Warhade**")
    st.markdown("---")

    # ── METRIC CARD HELPER ────────────────────────────────────────────────────
    def metric_card(value, label, bg):
        return (
            f'<div style="flex:1; min-width:150px; background:{bg}; border-radius:12px;'
            f'padding:18px 16px; text-align:center; color:#ffffff;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.35);">'
            f'<div style="font-size:26px; font-weight:800; letter-spacing:-0.5px;">{value}</div>'
            f'<div style="font-size:11px; margin-top:5px; opacity:0.88;'
            f'text-transform:uppercase; letter-spacing:0.8px;">{label}</div>'
            f'</div>'
        )

    def info_card(value, label, bg):
        """Smaller card for inline stats inside tabs"""
        return (
            f'<div style="flex:1; min-width:130px; background:{bg}; border-radius:10px;'
            f'padding:14px 12px; text-align:center; color:#ffffff;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.3);">'
            f'<div style="font-size:20px; font-weight:700;">{value}</div>'
            f'<div style="font-size:10px; margin-top:4px; opacity:0.85;'
            f'text-transform:uppercase; letter-spacing:0.7px;">{label}</div>'
            f'</div>'
        )

    # ── METRICS: show 0 before load, real numbers after ───────────────────────
    if not st.session_state.dataset_loaded or st.session_state.df is None:
        # Show zeroed-out cards + prominent load prompt
        st.markdown(
            '<div style="display:flex; gap:10px; margin:12px 0 20px 0; flex-wrap:wrap;">'
            + metric_card("—", "Total Games",      "#37474F")
            + metric_card("—", "Platforms",         "#37474F")
            + metric_card("—", "Genres",            "#37474F")
            + metric_card("—", "Avg Review Score",  "#37474F")
            + metric_card("—", "Total Players",     "#37474F")
            + '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        # Big central prompt
        st.markdown(
            '<div style="text-align:center; padding:60px 20px;">'
            '<div style="font-size:48px;">🎮</div>'
            '<div style="font-size:24px; font-weight:700; margin:12px 0 8px;">No Dataset Loaded</div>'
            '<div style="font-size:15px; color:#888; margin-bottom:20px;">'
            'Use the sidebar to load a dataset and unlock all 5 analytics tabs.</div>'
            '<div style="font-size:13px; color:#666;">'
            '👈 Click <strong>Load Default Dataset</strong> to start instantly<br>'
            'or upload your own CSV file below it.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        return   # Stop rendering — nothing else to show without data

    # ── DATA IS LOADED — proceed normally ─────────────────────────────────────
    df = st.session_state.df
    conn = build_database(df)

    # Update header with real counts
    label_tag = "Default Dataset" if st.session_state.dataset_label == "default" else "Custom Dataset"
    st.markdown(
        f"**{len(df)} games · {df['platform'].nunique()} platforms · "
        f"ML-powered · {label_tag} · Built by Kartikeya Warhade**"
    )

    # Real metric cards
    total_games   = len(df)
    total_plat    = df['platform'].nunique()
    total_genres  = df['genre'].nunique()
    avg_score     = df['review_score'].mean()
    total_players = int(df['player_count_millions'].sum())

    st.markdown(
        '<div style="display:flex; gap:10px; margin:12px 0 20px 0; flex-wrap:wrap;">'
        + metric_card(f"{total_games:,}",    "Total Games",      "#1565C0")
        + metric_card(str(total_plat),        "Platforms",        "#0277BD")
        + metric_card(str(total_genres),      "Genres",           "#00695C")
        + metric_card(f"{avg_score:.2f}/5.0", "Avg Review Score", "#2E7D32")
        + metric_card(f"{total_players:,}M",  "Total Players",    "#BF360C")
        + '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Analytics",
        "🎯 Game Recommender",
        "💬 Sentiment Analysis",
        "🤖 ML Predictor",
        "🔥 Trending Games"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — MARKET ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Market Analytics Dashboard")
        st.markdown("All charts powered by **SQL queries** on a live SQLite database.")

        # Platform breakdown
        st.subheader("Games by Platform")
        platform_df = pd.read_sql_query("""
            SELECT platform, COUNT(*) as total_games,
                   ROUND(AVG(review_score),2) as avg_score,
                   ROUND(AVG(indian_popularity),2) as avg_india_pop
            FROM games GROUP BY platform ORDER BY total_games DESC
        """, conn)

        col1, col2 = st.columns(2)
        with col1:
            fig = make_pie_chart(platform_df['platform'].tolist(),
                                 platform_df['total_games'].tolist(),
                                 "Games Distribution by Platform")
            st.pyplot(fig); plt.close()
        with col2:
            fig = make_bar_chart(platform_df, 'platform', 'avg_score',
                                 "Average Review Score by Platform", color='#0288D1')
            st.pyplot(fig); plt.close()

        st.dataframe(platform_df, use_container_width=True, hide_index=True)

        # Genre analysis
        st.subheader("Genre Analysis")
        genre_df = pd.read_sql_query("""
            SELECT genre, COUNT(*) as game_count,
                   ROUND(AVG(review_score),2) as avg_score,
                   ROUND(AVG(player_count_millions),1) as avg_players_m,
                   ROUND(AVG(indian_popularity),1) as india_popularity
            FROM games GROUP BY genre ORDER BY game_count DESC LIMIT 15
        """, conn)

        col1, col2 = st.columns(2)
        with col1:
            fig = make_bar_chart(genre_df.head(10), 'genre', 'game_count',
                                 "Top 10 Genres by Count", color='#1565C0')
            st.pyplot(fig); plt.close()
        with col2:
            fig = make_bar_chart(
                genre_df.sort_values('india_popularity',ascending=False).head(10),
                'genre', 'india_popularity',
                "Top 10 Genres by Indian Popularity", color='#2E7D32')
            st.pyplot(fig); plt.close()

        st.dataframe(genre_df, use_container_width=True, hide_index=True)

        # Top Indian games
        st.subheader("Top Games by Indian Popularity")
        india_df = pd.read_sql_query("""
            SELECT name, genre, platform, review_score,
                   indian_popularity, player_count_millions
            FROM games ORDER BY indian_popularity DESC, review_score DESC LIMIT 20
        """, conn)

        col1, col2 = st.columns(2)
        with col1:
            fig = make_bar_chart(india_df.head(10), 'name', 'indian_popularity',
                                 "Top 10 — Indian Market Popularity",
                                 color='#E65100', horizontal=True)
            st.pyplot(fig); plt.close()
        with col2:
            plat_counts = india_df['platform'].value_counts().reset_index()
            plat_counts.columns = ['platform','count']
            fig = make_pie_chart(plat_counts['platform'].tolist(),
                                 plat_counts['count'].tolist(),
                                 "Platform Share — Top 20 Indian Games")
            st.pyplot(fig); plt.close()

        st.dataframe(india_df, use_container_width=True, hide_index=True)

        # Release year trends
        st.subheader("Game Releases by Year")
        year_df = pd.read_sql_query("""
            SELECT release_year, COUNT(*) as games_released,
                   ROUND(AVG(review_score),2) as avg_score
            FROM games WHERE release_year >= 2010
            GROUP BY release_year ORDER BY release_year
        """, conn)

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.bar(year_df['release_year'], year_df['games_released'],
                color='#1565C0', alpha=0.7, label='Games Released')
        ax2.plot(year_df['release_year'], year_df['avg_score'],
                 color='#E65100', marker='o', linewidth=2, label='Avg Score')
        ax1.set_xlabel("Year"); ax1.set_ylabel("Games Released", color='#1565C0')
        ax2.set_ylabel("Avg Review Score", color='#E65100')
        ax1.set_title("Games Released Per Year vs Average Review Score",
                      fontsize=12, fontweight='bold')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — GAME RECOMMENDER (FIX 1 + FIX 2)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("Game Recommendation Engine")
        st.markdown(
            "Uses **Cosine Similarity** to find games most similar to your selection. "
            "The slider maximum dynamically adjusts to show only genuinely similar games."
        )

        with st.spinner("Building similarity matrix..."):
            sim_matrix = build_similarity_matrix(df)

        col1, col2 = st.columns([1, 2])

        with col1:
            platform_for_filter = st.selectbox(
                "🎮 Select Platform",
                ["All","Mobile","PC","PlayStation","Xbox","Nintendo"],
                help="Filter game list by platform"
            )

            rec_platform = st.radio(
                "Recommend from which platform?",
                ["Same Platform Only","All Platforms"]
            )

        with col2:
            if platform_for_filter == "All":
                game_options = sorted(df['name'].unique().tolist())
            else:
                game_options = sorted(
                    df[df['platform'] == platform_for_filter]['name'].unique().tolist()
                )
            selected_game = st.selectbox("🔍 Select a Game", game_options)

        # ── FIX 1: Dynamic slider based on actual similar game count ──────────
        if selected_game:
            game_platform = df[df['name'] == selected_game]['platform'].values[0]
            filter_plat = game_platform if rec_platform == "Same Platform Only" else "All"

            # Get ALL similar games first (threshold 0.75 to be generous)
            all_similar = get_all_similar_games(
                df, sim_matrix, selected_game, filter_plat, threshold=0.75
            )
            total_similar = len(all_similar)

            if total_similar == 0:
                st.warning("No similar games found above the 75% similarity threshold. Try 'All Platforms'.")
                n_recs = 0
            else:
                st.markdown(
                    f"**{total_similar} similar games found** for *{selected_game}* "
                    f"(≥75% similarity, {filter_plat} platform filter)"
                )
                # Slider: min 1, max = total_similar found
                n_recs = st.slider(
                    "How many recommendations to show?",
                    min_value=1,
                    max_value=total_similar,
                    value=min(8, total_similar),
                    help=f"There are {total_similar} games similar to {selected_game}. "
                         f"Drag to see more or fewer."
                )

        if st.button("🎯 Get Recommendations", type="primary"):
            if total_similar == 0:
                st.warning("No recommendations available.")
            else:
                recs = all_similar.head(n_recs).reset_index(drop=True)
                recs.index = recs.index + 1

                # Game info card — HTML so it works in dark mode
                game_info = df[df['name'] == selected_game].iloc[0]
                st.markdown(f"### Selected: **{selected_game}**")
                st.markdown(
                    '<div style="display:flex; gap:10px; margin:10px 0 16px 0; flex-wrap:wrap;">'
                    + info_card(game_info['genre'],                    "Genre",             "#1565C0")
                    + info_card(game_info['platform'],                  "Platform",          "#0277BD")
                    + info_card(f"{game_info['review_score']}/5.0",    "Review Score",      "#2E7D32")
                    + info_card(f"{int(game_info['indian_popularity'])}/10", "India Popularity", "#BF360C")
                    + '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"### Top {n_recs} of {total_similar} Similar Games")

                # Colour similarity score cells
                def color_sim(val):
                    if val >= 0.95: return 'background-color: #1B5E20; color: #ffffff; font-weight:bold'
                    elif val >= 0.85: return 'background-color: #E65100; color: #ffffff'
                    else: return 'background-color: #B71C1C; color: #ffffff'

                st.dataframe(
                    recs.style.applymap(color_sim, subset=['similarity_score']),
                    use_container_width=True
                )

                # Bar chart
                fig, ax = plt.subplots(figsize=(9, max(3, n_recs * 0.45)))
                bar_colors = [
                    '#1B5E20' if x >= 0.95 else
                    '#F9A825' if x >= 0.85 else '#BF360C'
                    for x in recs['similarity_score']
                ]
                ax.barh(recs['name'][::-1], recs['similarity_score'][::-1],
                        color=bar_colors[::-1], edgecolor='white')
                ax.set_xlabel("Similarity Score")
                ax.set_title(f"Similarity Scores — {selected_game}", fontsize=11, fontweight='bold')
                ax.set_xlim(0.70, 1.05)
                ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.6, label='≥0.95 Very Similar')
                ax.axvline(x=0.85, color='orange', linestyle='--', alpha=0.6, label='≥0.85 Similar')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

        # FIX 2: "How Cosine Similarity Works" expander REMOVED

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — SENTIMENT ANALYSIS (FIX 4: dark backgrounds)
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("Sentiment Analysis Engine")
        st.markdown(
            "**VADER NLP** analyses game reviews and classifies them as "
            "Positive 🟢, Negative 🔴, or Neutral 🟠. "
            "Compound score: +1.0 = most positive, -1.0 = most negative."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Sample Reviews")
            game_choice = st.selectbox("Choose a game", list(SAMPLE_REVIEWS.keys()))

            if st.button("🔬 Analyse Reviews", type="primary"):
                reviews = SAMPLE_REVIEWS[game_choice]
                results_df = analyze_sentiment(reviews)

                with col2:
                    st.subheader(f"Results — {game_choice}")

                    pos = (results_df['Sentiment']=='Positive').sum()
                    neg = (results_df['Sentiment']=='Negative').sum()
                    neu = (results_df['Sentiment']=='Neutral').sum()
                    avg_c = results_df['Compound'].mean()

                    st.markdown(
                        '<div style="display:flex; gap:10px; margin:10px 0 16px 0; flex-wrap:wrap;">'
                        + info_card(str(pos),        "Positive",    "#1B5E20")
                        + info_card(str(neg),        "Negative",    "#B71C1C")
                        + info_card(str(neu),        "Neutral",     "#E65100")
                        + info_card(f"{avg_c:.3f}",  "Avg Compound","#37474F")
                        + '</div>',
                        unsafe_allow_html=True
                    )

                    # FIX 4: Dark background HTML cards — easy to read
                    st.markdown("#### Review Results")
                    for _, row in results_df.iterrows():
                        css_class = {
                            'Positive': 'sentiment-positive',
                            'Negative': 'sentiment-negative',
                            'Neutral':  'sentiment-neutral'
                        }[row['Sentiment']]
                        emoji = {'Positive':'😊','Negative':'😞','Neutral':'😐'}[row['Sentiment']]
                        st.markdown(
                            f'<div class="{css_class}">'
                            f'{emoji} <strong>{row["Sentiment"]}</strong> '
                            f'(Score: {row["Compound"]}) — {row["Review"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")
                    fig = make_pie_chart(
                        ['Positive','Negative','Neutral'], [pos,neg,neu],
                        f"Sentiment Distribution — {game_choice}"
                    )
                    st.pyplot(fig); plt.close()

                    # Bar chart of compound scores
                    fig2, ax = plt.subplots(figsize=(9, 4))
                    bar_c = ['#2E7D32' if s>=0.05 else '#C62828' if s<=-0.05 else '#E65100'
                             for s in results_df['Compound']]
                    bars = ax.bar(range(1, len(results_df)+1),
                                  results_df['Compound'], color=bar_c, edgecolor='white')
                    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
                    ax.axhline(0.05, color='green', linewidth=0.6, linestyle=':', alpha=0.6)
                    ax.axhline(-0.05, color='red', linewidth=0.6, linestyle=':', alpha=0.6)
                    ax.set_xlabel("Review Number")
                    ax.set_ylabel("Compound Score")
                    ax.set_title("Compound Score per Review", fontsize=11, fontweight='bold')
                    ax.set_ylim(-1.1, 1.1)
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close()

        # Custom review analyser
        st.markdown("---")
        st.subheader("Analyse Your Own Review")
        custom_review = st.text_area(
            "Type any game review:",
            placeholder="e.g. This game is absolutely incredible!"
        )

        if st.button("Analyse My Review") and custom_review.strip():
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(custom_review)
            compound = scores['compound']

            if compound >= 0.05:
                sentiment, css = "Positive 😊", "sentiment-positive"
            elif compound <= -0.05:
                sentiment, css = "Negative 😞", "sentiment-negative"
            else:
                sentiment, css = "Neutral 😐", "sentiment-neutral"

            st.markdown(
                f'<div class="{css}" style="font-size:16px; padding:12px;">'
                f'<strong>Sentiment: {sentiment}</strong> &nbsp;|&nbsp; '
                f'Compound Score: {compound:.3f}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="display:flex; gap:10px; margin:10px 0; flex-wrap:wrap;">'
                + info_card(f"{scores['pos']:.3f}", "Positive Score", "#1B5E20")
                + info_card(f"{scores['neg']:.3f}", "Negative Score", "#B71C1C")
                + info_card(f"{scores['neu']:.3f}", "Neutral Score",  "#E65100")
                + '</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — ML PREDICTOR (FIX 5: year extended to 2030)
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("Random Forest — Game Success Predictor")
        st.markdown("""
        **Predicts** whether a game will have high Indian popularity (score ≥ 8/10).
        **Algorithm:** 100 decision trees, 5-fold cross-validation.
        **Works for future games too** — enter 2027, 2028, 2029, 2030 release years
        and the model predicts based on the patterns it learned from existing games.
        """)

        with st.spinner("Training Random Forest..."):
            rf_model, accuracy, cv_scores, importances, X_test, y_test, y_pred = train_random_forest(df)

        if rf_model is None:
            st.error("Not enough class diversity in the current dataset to train. Try uploading a larger CSV.")
        else:
            st.markdown(
                '<div style="display:flex; gap:10px; margin:10px 0 20px 0; flex-wrap:wrap;">'
                + info_card(f"{accuracy*100:.1f}%",          "Test Accuracy",      "#1565C0")
                + info_card(f"{cv_scores.mean()*100:.1f}%",  "CV Mean Accuracy",   "#2E7D32")
                + info_card(f"±{cv_scores.std()*100:.1f}%",  "CV Std Dev",         "#00695C")
                + info_card("100",                            "Trees in Forest",    "#37474F")
                + '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Feature Importance")
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_fi = ['#1565C0' if i==0 else '#0288D1' if i<3 else '#90CAF9'
                             for i in range(len(importances))]
                ax.barh(importances.index[::-1], importances.values[::-1],
                        color=colors_fi[::-1], edgecolor='white')
                ax.set_xlabel("Importance Score")
                ax.set_title("What Drives Indian Gaming Success?",
                             fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col2:
                st.subheader("5-Fold Cross-Validation")
                fig, ax = plt.subplots(figsize=(7, 5))
                bar_cv = ['#2E7D32' if s>=0.75 else '#F9A825' for s in cv_scores]
                ax.bar([f"Fold {i+1}" for i in range(5)], cv_scores*100,
                       color=bar_cv, edgecolor='white')
                ax.axhline(cv_scores.mean()*100, color='red', linestyle='--',
                           linewidth=2, label=f'Mean: {cv_scores.mean()*100:.1f}%')
                ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
                ax.set_title("CV Accuracy per Fold", fontsize=12, fontweight='bold')
                ax.legend(); ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            # ── PREDICT NEW GAME ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Predict Success of Any Game — Including Future Releases")
            st.markdown(
                "Enter the expected features of an upcoming or hypothetical game. "
                "Release years up to **2030** are supported — the model extrapolates "
                "from patterns in existing data to predict future game success."
            )

            pc1,pc2,pc3 = st.columns(3)
            with pc1:
                p_multiplayer    = st.selectbox("Multiplayer?", [1,0], format_func=lambda x:"Yes" if x else "No")
                p_battle_royale  = st.selectbox("Battle Royale?", [0,1], format_func=lambda x:"Yes" if x else "No")
                p_open_world     = st.selectbox("Open World?", [0,1], format_func=lambda x:"Yes" if x else "No")
                p_fps            = st.selectbox("FPS Shooter?", [0,1], format_func=lambda x:"Yes" if x else "No")
            with pc2:
                p_sports         = st.selectbox("Sports Game?", [0,1], format_func=lambda x:"Yes" if x else "No")
                p_rpg            = st.selectbox("RPG Elements?", [0,1], format_func=lambda x:"Yes" if x else "No")
                p_indie          = st.selectbox("Indie Game?", [0,1], format_func=lambda x:"Yes" if x else "No")
            with pc3:
                p_review  = st.slider("Expected Review Score", 1.0, 5.0, 4.0, 0.1)
                p_players = st.slider("Expected Players (Millions)", 1, 300, 30)
                # FIX 5: Year range extended to 2030
                p_year    = st.slider("Release Year", 2020, 2030, 2025)

            if p_year > 2025:
                st.info(
                    f"📅 Predicting for a **{p_year} future release**. "
                    "The model uses patterns from 312 existing games to estimate success probability."
                )

            if st.button("🤖 Predict Success", type="primary"):
                input_features = np.array([[
                    p_multiplayer, p_battle_royale, p_open_world, p_fps,
                    p_sports, p_rpg, p_indie, p_review, p_players, 5, p_year
                ]])

                prediction   = rf_model.predict(input_features)[0]
                probabilities = rf_model.predict_proba(input_features)[0]

                if prediction == 1:
                    st.success(
                        f"✅ **HIGH SUCCESS** predicted in Indian Market  "
                        f"— Confidence: {probabilities[1]*100:.1f}%"
                    )
                else:
                    st.warning(
                        f"⚠️ **MODERATE / LOW SUCCESS** predicted in Indian Market  "
                        f"— Confidence: {probabilities[0]*100:.1f}%"
                    )

                st.markdown(
                    '<div style="display:flex; gap:10px; margin:10px 0; flex-wrap:wrap;">'
                    + info_card(f"{probabilities[1]*100:.1f}%", "High Success Probability", "#2E7D32")
                    + info_card(f"{probabilities[0]*100:.1f}%", "Low Success Probability",  "#BF360C")
                    + '</div>',
                    unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — TRENDING GAMES
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("Trending Games Dashboard")

        t1,t2,t3 = st.columns(3)
        with t1:
            trend_platform = st.selectbox("Platform",
                ["All","Mobile","PC","PlayStation","Xbox","Nintendo"], key="tp")
        with t2:
            trend_genre = st.selectbox("Genre",
                ["All"] + sorted(df['genre'].unique().tolist()), key="tg")
        with t3:
            rank_by = st.selectbox("Rank By",
                ["indian_popularity","review_score","player_count_millions"],
                format_func=lambda x: {
                    "indian_popularity":"Indian Popularity",
                    "review_score":"Review Score",
                    "player_count_millions":"Player Count (Millions)"
                }[x])

        where_clauses = []
        if trend_platform != "All":
            where_clauses.append(f"platform = '{trend_platform}'")
        if trend_genre != "All":
            where_clauses.append(f"genre = '{trend_genre}'")
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        trending_df = pd.read_sql_query(f"""
            SELECT name, genre, platform, review_score,
                   player_count_millions, indian_popularity, release_year, developer
            FROM games {where_sql}
            ORDER BY {rank_by} DESC LIMIT 20
        """, conn)

        st.markdown(f"**{len(trending_df)} results shown**")
        st.dataframe(trending_df, use_container_width=True, hide_index=True)

        if len(trending_df) > 0:
            col1,col2 = st.columns(2)
            with col1:
                fig = make_bar_chart(trending_df.head(10), 'name', rank_by,
                                     f"Top 10 — {rank_by.replace('_',' ').title()}",
                                     color='#1565C0', horizontal=True)
                st.pyplot(fig); plt.close()
            with col2:
                if trend_platform == "All":
                    pd_dist = trending_df['platform'].value_counts().reset_index()
                    pd_dist.columns = ['platform','count']
                    fig = make_pie_chart(pd_dist['platform'].tolist(),
                                         pd_dist['count'].tolist(),
                                         "Platform Distribution")
                else:
                    gd_dist = trending_df['genre'].value_counts().reset_index()
                    gd_dist.columns = ['genre','count']
                    fig = make_pie_chart(gd_dist['genre'].head(6).tolist(),
                                         gd_dist['count'].head(6).tolist(),
                                         "Genre Distribution")
                st.pyplot(fig); plt.close()

        # Indian market insights
        st.markdown("---")
        st.subheader("Indian Gaming Market Intelligence")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Genres by Indian Popularity**")
            st.caption(
                "Each row is one game scoring ≥ 8/10 on Indian popularity. "
                "'Rank in Genre' shows this game's position within its genre — "
                "Rank 1 = most popular game in that genre by India Score."
            )

            i1_base = pd.read_sql_query("""
                SELECT genre, COUNT(*) as game_count,
                       ROUND(AVG(indian_popularity),2) as avg_india_score
                FROM games WHERE indian_popularity >= 8
                GROUP BY genre ORDER BY avg_india_score DESC LIMIT 10
            """, conn)

            genre_game_rows = []
            for _, row in i1_base.iterrows():
                games_in_genre = df[
                    (df["genre"] == row["genre"]) &
                    (df["indian_popularity"] >= 8)
                ].sort_values(
                    ["indian_popularity", "review_score"], ascending=False
                )[["name", "platform", "indian_popularity", "review_score"]]

                for rank, (_, gr) in enumerate(games_in_genre.iterrows(), start=1):
                    genre_game_rows.append({
                        "Genre":         row["genre"],
                        "Game Name":     gr["name"],
                        "Platform":      gr["platform"],
                        "India Score":   gr["indian_popularity"],
                        "Review Score":  gr["review_score"],
                        "Genre Avg":     row["avg_india_score"],
                        "Rank in Genre": rank,
                    })

            i1_expanded = pd.DataFrame(genre_game_rows)
            st.dataframe(i1_expanded, use_container_width=True, hide_index=True)

            i1_expanded = pd.DataFrame(genre_game_rows)
            st.dataframe(i1_expanded, use_container_width=True, hide_index=True)

            with st.expander("📋 Genre Summary — one row per genre"):
                st.dataframe(
                    i1_base.rename(columns={
                        "game_count":      "Games Scoring ≥8/10",
                        "avg_india_score": "Avg India Score"
                    }),
                    use_container_width=True, hide_index=True
                )

        with col2:
            st.markdown("**Platform Success Rate in India**")
            st.caption("'High Pop Games' = games scoring ≥ 8/10 Indian popularity on that platform.")
            i2 = pd.read_sql_query("""
                SELECT platform,
                       SUM(CASE WHEN indian_popularity>=8 THEN 1 ELSE 0 END) as high_pop_games,
                       COUNT(*) as total_games,
                       ROUND(100.0*SUM(CASE WHEN indian_popularity>=8 THEN 1 ELSE 0 END)/COUNT(*),1) as success_rate_pct
                FROM games GROUP BY platform ORDER BY success_rate_pct DESC
            """, conn)
            st.dataframe(i2, use_container_width=True, hide_index=True)

            # Visual bar chart of success rates
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bar_c = ['#1565C0','#0288D1','#00838F','#2E7D32','#558B2F']
            ax.barh(i2['platform'][::-1], i2['success_rate_pct'][::-1],
                    color=bar_c[:len(i2)][::-1], edgecolor='white')
            ax.set_xlabel("Success Rate (%)")
            ax.set_title("Platform Success Rate in India", fontsize=11, fontweight='bold')
            ax.axvline(x=25, color='orange', linestyle='--', alpha=0.6, label='25% benchmark')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig); plt.close()


if __name__ == "__main__":
    main()
