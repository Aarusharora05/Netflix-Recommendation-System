# 🍿 Netflix Recommendation System

A content-based movie and TV show recommendation engine built with Python and Streamlit, powered by TF-IDF vectorization and cosine similarity.

---

## 🚀 Demo

> Enter any title from the Netflix dataset and get 10 personalized recommendations instantly.

---

## 📌 Features

- 🔍 **Content-Based Filtering** — recommends titles based on director, cast, genre, and description
- ⚡ **Fast & Cached** — uses Streamlit's `@st.cache_data` and `@st.cache_resource` for instant repeat queries
- 🎛️ **Interactive UI** — searchable dropdown with expandable result cards showing genre and overview
- 📊 **TF-IDF + Cosine Similarity** — NLP-powered similarity scoring across the full Netflix catalog

---

## 🧠 How It Works

1. **Data Loading** — reads `netflix_titles.csv` (8,800+ titles)
2. **Feature Engineering** — combines `director`, `cast`, `listed_in`, and `description` into a single text "soup"
3. **Vectorization** — applies `TfidfVectorizer` (with English stop-words removed) to convert text to numerical features
4. **Similarity Scoring** — computes a cosine similarity matrix across all titles using `linear_kernel`
5. **Recommendation** — given a title, retrieves the top 10 most similar titles by similarity score

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Streamlit |
| Data Processing | Pandas |
| ML / NLP | scikit-learn (TF-IDF, cosine similarity) |
| Dataset | Netflix Movies and TV Shows (Kaggle) |

---

## 📂 Project Structure

```
Netflix-Recommendation-System/
├── app.py                  # Main Streamlit application
├── test.py                 # Unit tests
├── netflix_titles.csv      # Netflix dataset
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Aarusharora05/Netflix-Recommendation-System.git
cd Netflix-Recommendation-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📋 Requirements

```
streamlit
pandas
scikit-learn
```

---

## 📸 Usage

1. Launch the app with `streamlit run app.py`
2. Use the dropdown to search for or select a Netflix title
3. Click **Show Recommendations**
4. Expand any result card to view its genre and description

---

## 🔮 Future Improvements

- [ ] Add collaborative filtering using user ratings
- [ ] Integrate with the TMDB API for posters and trailers
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add filters by genre, release year, and content type (Movie vs TV Show)

---

## 📄 Dataset

The dataset used is the [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset from Kaggle, containing over 8,800 titles with metadata including cast, director, genre, and description.

---

## 👤 Author

**Aarush Arora**  
[GitHub](https://github.com/Aarusharora05)
