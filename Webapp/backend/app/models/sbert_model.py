import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from .base_model import BaseRecommendationModel


class SBERTModel(BaseRecommendationModel):
    """SBERT-based movie recommendation model."""
    
    def __init__(self):
        super().__init__(
            name="Overview + Genre Model",
            description="SBERT on overview + genre filtering (cosine similarity)",
            author="Brayan Torres"
        )
        self.df = None
        self.model = None
        self.overview_embeddings = None
    
    def load_model(self):
        """Load the SBERT model and movie data."""
        # Load dataset
        self.df = pd.read_csv("imdb_top_1000.csv")
        self.df = self.df.dropna(subset=["Overview", "Genre"]).reset_index(drop=True)
        
        # Load SBERT model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Encode all plot overviews
        self.overview_embeddings = self.model.encode(
            self.df["Overview"].tolist(),
            convert_to_tensor=True
        )
    
    def recommend(self, movie_title: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generate movie recommendations using SBERT and genre similarity."""
        if self.df is None or self.model is None:
            self.load_model()
        
        title_idx = self.df[self.df["Series_Title"].str.lower() == movie_title.lower()].index
        if len(title_idx) == 0:
            return []
        
        title_idx = title_idx[0]
        input_genres = set(self.df.iloc[title_idx]["Genre"].lower().split(", "))
        query_embedding = self.overview_embeddings[title_idx]
        cos_scores = util.cos_sim(query_embedding, self.overview_embeddings)[0]
        top_results = cos_scores.argsort(descending=True)
        
        results = []
        for idx_tensor in top_results:
            idx = int(idx_tensor)
            if idx == title_idx:
                continue
            
            movie_genres = set(self.df.iloc[idx]["Genre"].lower().split(", "))
            if input_genres & movie_genres:
                results.append({
                    "title": self.df.iloc[idx]["Series_Title"],
                    "genre": self.df.iloc[idx]["Genre"],
                    "similarity": float(cos_scores[idx]),
                    "rating": float(self.df.iloc[idx]["IMDB_Rating"]),
                    "overview": self.df.iloc[idx]["Overview"]
                })
            
            if len(results) >= num_recommendations:
                break
        
        return results 