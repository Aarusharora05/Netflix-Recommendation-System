import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")

# --- 1. LOAD DATA (Cached) ---
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('netflix_titles.csv')
        return df
    except FileNotFoundError:
        return None

# --- 2. PREPROCESS & BUILD MODEL (Cached) ---
@st.cache_resource
def build_model(df):
    # Data Cleaning
    relevant_columns = ['title', 'director', 'cast', 'listed_in', 'description']
    df = df[relevant_columns].fillna('')
    
    # Create "Soup"
    df['combined_features'] = (
        df['director'] + ' ' + 
        df['cast'] + ' ' + 
        df['listed_in'] + ' ' + 
        df['description']
    )
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Cosine Similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

# --- 3. RECOMMENDATION FUNCTION ---
def get_recommendations(title, df, cosine_sim):
    # Get index of the movie
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in indices:
        return []

    idx = indices[title]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Top 10 movies (exclude the first one which is the movie itself)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    # Return titles and descriptions
    return df.iloc[movie_indices][['title', 'description', 'listed_in']]

# --- MAIN APP UI ---
def main():
    # Sidebar
    st.sidebar.title("🎬 Control Panel")
    st.sidebar.write("Select a movie you like, and we'll suggest 10 more!")

    # Main Title
    st.title("🍿 Netflix Recommendation System")
    st.markdown("### Find your next favorite movie using Machine Learning")
    
    # Load Data
    raw_df = load_data()
    
    if raw_df is not None:
        df, cosine_sim = build_model(raw_df)
        
        # Movie Selection Dropdown
        # We limit the list to the first 5000 to keep the dropdown fast, or use all
        movie_list = df['title'].values
        selected_movie = st.selectbox(
            "Type or select a movie from the dataset:", 
            movie_list
        )
        
        # Recommendation Button
        if st.button('Show Recommendations'):
            with st.spinner('Finding similar movies...'):
                recommendations = get_recommendations(selected_movie, df, cosine_sim)
            
            if len(recommendations) > 0:
                st.success(f"Because you liked **{selected_movie}**:")
                
                # Display results in a grid
                for index, row in recommendations.iterrows():
                    with st.expander(f"🎥 {row['title']}"):
                        st.write(f"**Genre:** {row['listed_in']}")
                        st.write(f"**Overview:** {row['description']}")
            else:
                st.warning("Could not find recommendations.")
                
    else:
        st.error("Dataset 'netflix_titles.csv' not found! Please ensure it is in the same folder.")

if __name__ == "__main__":
    main()