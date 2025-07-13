# AI-powered-movie-recommendation-system-title
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load movie dataset
@st.cache_data
def load_data():
    # Sample dataset (replace with larger CSV if needed)
    data = {
        'title': [
            'The Dark Knight', 'Inception', 'Interstellar', 'The Prestige',
            'Memento', 'Tenet', 'Dunkirk', 'Batman Begins', 'Insomnia'
        ],
        'description': [
            'Batman raises the stakes in his war on crime.',
            'A thief who steals corporate secrets through dream-sharing.',
            'A team travels through a wormhole in space.',
            'Two stage magicians engage in a rivalry.',
            'A man with short-term memory loss attempts to find his wife’s killer.',
            'Protagonist fights for the survival of the world through time inversion.',
            'Allied soldiers are evacuated during a fierce battle.',
            'The origin story of Batman.',
            'A detective investigates a murder in a small town.'
        ]
    }
    return pd.DataFrame(data)

# Load data
df = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get movie recommendations
def get_recommendations(title, df=df, cosine_sim=cosine_sim):
    title = title.lower()
    indices = pd.Series(df.index, index=df['title'].str.lower())
    if title not in indices:
        return ["Movie not found in database."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit Web App
st.title("🎬 AI-Powered Movie Recommender")
st.write("Enter a movie title you like, and get similar recommendations!")

movie_input = st.text_input("Movie Title", "Inception")

if st.button("Recommend"):
    recommendations = get_recommendations(movie_input)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
