import os
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from .base_model import BaseRecommendationModel

class MovieFeatureizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", a=1.0, b=5.0, svd_dims=50):
        self.model_name = model_name
        self.a = a
        self.b = b
        self.svd_dims = svd_dims

    def fit(self, X: pd.DataFrame, y=None):
        self.sbert = SentenceTransformer(self.model_name)
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,3))
        dirs = X["Director"].str.lower().str.replace(f"[{re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')}]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        self.tfidf.fit(dirs)
        self.svd = TruncatedSVD(self.svd_dims, random_state=42)
        self.svd.fit(self.tfidf.transform(dirs))
        return self

    def transform(self, X: pd.DataFrame):
        over = X["Overview"].str.lower().str.replace(f"[{re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')}]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        dirs = X["Director"].str.lower().str.replace(f"[{re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')}]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        texts = (over + " [SEP] director: " + dirs).tolist()
        emb = self.sbert.encode(texts, show_progress_bar=False)
        emb = normalize(emb) * self.a
        D = self.tfidf.transform(dirs)
        D = self.svd.transform(D)
        D = normalize(D) * self.b
        return np.hstack([emb, D])

class EugeneModel(BaseRecommendationModel):
    def __init__(self):
        super().__init__(
            name="Eugene's SBERT+Director Model",
            description="SBERT on overview + TFIDF+SVD on director, KNN",
            author="Eugene"
        )
        self.df = None
        self.featureizer = None
        self.knn = None

    def load_model(self):
        csv_path = os.path.join(os.path.dirname(__file__), '../../imdb_top_1000.csv')
        self.df = pd.read_csv(csv_path).dropna(subset=["Overview","Director"]).reset_index(drop=True)
        self.featureizer = MovieFeatureizer()
        feats = self.featureizer.fit_transform(self.df)
        self.knn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
        self.knn.fit(feats)

    def recommend(self, movie_title: str, num_recommendations: int = 5):
        if self.df is None or self.knn is None:
            self.load_model()
        mask = self.df["Series_Title"].str.lower() == movie_title.lower()
        if not mask.any():
            return []
        i = mask.idxmax()
        qf = self.featureizer.transform(self.df.loc[[i]])
        nbrs = self.knn.kneighbors(qf, n_neighbors=num_recommendations+1, return_distance=True)
        dists, idxs = nbrs
        rec_idxs = idxs[0][1:]
        rec_dists = dists[0][1:]
        results = []
        for ridx, dist in zip(rec_idxs, rec_dists):
            row = self.df.iloc[ridx]
            results.append({
                "title": row["Series_Title"],
                "genre": row["Genre"],
                "similarity": 1 - float(dist),
                "rating": float(row["IMDB_Rating"]),
                "overview": row["Overview"]
            })
        return results 