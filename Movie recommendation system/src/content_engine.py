
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ContentEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self.movie_ids = None

    def fit(self, movies):
        self.movie_ids = movies['movie_id'].values
        tfidf_matrix = self.vectorizer.fit_transform(movies['genres'].str.replace('|', ' '))
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def save(self, vectorizer_path, sim_matrix_path):
        with open(vectorizer_path, 'wb') as f: pickle.dump(self.vectorizer, f)
        with open(sim_matrix_path, 'wb') as f: pickle.dump(self.similarity_matrix, f)
