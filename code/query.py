from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import polars as pl
from data import movie_ids, titles, corpus

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def recommend_movies(query:str, top_k:int=10) -> pl.DataFrame:
    print("Preparing recommendation...")
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return pl.DataFrame({"movieId": [movie_ids[i] for i in top_indices],
                         "title": [titles[i] for i in top_indices],
                         "similarity": [float(similarities[i]) for i in top_indices]
                         })