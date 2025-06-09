import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommendationModel

class PatrickModel(BaseRecommendationModel):
    """Patrick's movie recommendation model using TF-IDF and genre similarity."""
    
    def __init__(self):
        super().__init__(
            name="Overview+Genre (TF-IDF)",
            description="TF-IDF on overview + one-hot genre, combined similarity",
            author="Patrick"
        )
        self.df = None
        self.tfidf = None
        self.mlb = None
        self.overview_tfidf = None
        self.genre_matrix = None
        self.combined_sim = None
    
    def load_model(self):
        """Load the model and compute similarities."""
        # Load dataset
        csv_path = os.path.join(os.path.dirname(__file__), '../../imdb_top_1000.csv')
        self.df = pd.read_csv(csv_path)
        
        # Clean overviews
        self.df['clean_overview'] = self.df['Overview'].fillna('').str.lower()
        self.df['clean_overview'] = self.df['clean_overview'].str.replace(r'[^\w\s]', ' ', regex=True)
        self.df['clean_overview'] = self.df['clean_overview'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Process genres
        self.df['genre_list'] = self.df['Genre'].str.split(',\s*')
        
        # TF-IDF on overviews
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.overview_tfidf = self.tfidf.fit_transform(self.df['clean_overview'])
        
        # One-hot encode genres
        self.mlb = MultiLabelBinarizer()
        self.genre_matrix = self.mlb.fit_transform(self.df['genre_list'])
        
        # Compute similarities
        overview_sim = cosine_similarity(self.overview_tfidf)
        genre_sim = cosine_similarity(self.genre_matrix)
        
        # Combine similarities (80% overview, 20% genre)
        alpha = 0.8
        self.combined_sim = alpha * overview_sim + (1 - alpha) * genre_sim
    
    def recommend(self, movie_title: str, num_recommendations: int = 5):
        """Generate movie recommendations using combined similarity."""
        if self.df is None or self.combined_sim is None:
            self.load_model()
        
        # Find movie index
        mask = self.df['Series_Title'].str.lower() == movie_title.lower()
        if not mask.any():
            return []
        
        idx = mask.idxmax()
        
        # Get similarity scores
        sim_scores = list(enumerate(self.combined_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the input movie)
        top_indices = [i for i, _ in sim_scores[1:num_recommendations+1]]
        
        # Format results
        results = []
        for i in top_indices:
            row = self.df.iloc[i]
            results.append({
                "title": row["Series_Title"],
                "genre": row["Genre"],
                "similarity": float(sim_scores[i][1]),
                "rating": float(row["IMDB_Rating"]),
                "overview": row["Overview"]
            })
        
        return results 