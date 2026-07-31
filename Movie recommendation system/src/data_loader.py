
import pandas as pd
import numpy as np
import os

def load_data():
    np.random.seed(42)
    # Synthetic data generation for demonstration to avoid long downloads
    users = pd.DataFrame({'user_id': range(1, 101)})
    movies = pd.DataFrame({
        'movie_id': range(1, 51),
        'title': [f'Movie {i}' for i in range(1, 51)],
        'genres': np.random.choice(['Action|Sci-Fi', 'Comedy', 'Drama|Romance', 'Horror', 'Documentary'], 50)
    })
    
    # 500 random interactions
    interactions = pd.DataFrame({
        'user_id': np.random.randint(1, 101, 500),
        'movie_id': np.random.randint(1, 51, 500),
        'rating': np.random.randint(1, 6, 500)
    }).drop_duplicates(subset=['user_id', 'movie_id'])
    
    return users, movies, interactions
