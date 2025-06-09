import os
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from .base_model import BaseRecommendationModel

class KyleModel(BaseRecommendationModel):
    def __init__(self):
        super().__init__(
            name="Kyle's KNN Hybrid Model",
            description="TF-IDF + genre + director + stars + numeric features (KNN)",
            author="Kyle"
        )
        self.df = None
        self.model = None
        self.X = None
        self.vectorizer = None
        self.genre_mlb = None
        self.dir_onehot = None
        self.star_mlb = None
        self.scaler = None

    def load_model(self):
        csv_path = os.path.join(os.path.dirname(__file__), '../../imdb_top_1000.csv')
        self.df = pd.read_csv(csv_path)
        df = self.df
        # Preprocess overview
        def preprocess(text):
            tokens = re.findall(r'\b\w+\b', str(text).lower())
            return " ".join([t for t in tokens if t not in ENGLISH_STOP_WORDS])
        df['filtered_overview'] = df['Overview'].fillna('').apply(preprocess)
        # TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_tfidf = self.vectorizer.fit_transform(df['filtered_overview'])
        # Genres
        self.genre_mlb = MultiLabelBinarizer(sparse_output=True)
        genre_mat = self.genre_mlb.fit_transform(df['Genre'].str.split(',').apply(lambda lst: [g.strip() for g in lst]))
        # Director
        self.dir_onehot = pd.get_dummies(df['Director'], prefix='dir', sparse=True)
        # Stars
        stars = df[['Star1','Star2','Star3','Star4']].apply(lambda col: col.str.strip())
        self.star_mlb = MultiLabelBinarizer(sparse_output=True)
        star_mat = self.star_mlb.fit_transform(stars.values.tolist())
        # Numeric features
        num = df[['IMDB_Rating','Meta_score','No_of_Votes']].fillna(0)
        self.scaler = StandardScaler(with_mean=False)
        num_mat = self.scaler.fit_transform(num)
        # Combine all features
        self.X = hstack([
            X_tfidf,
            genre_mat,
            self.dir_onehot.sparse.to_coo(),
            star_mat,
            num_mat
        ]).tocsr()
        # KNN model
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.X)

    def recommend(self, movie_title: str, num_recommendations: int = 5):
        if self.df is None or self.model is None:
            self.load_model()
        df = self.df
        mask = df['Series_Title'].str.lower() == movie_title.lower()
        if not mask.any():
            return []
        idx = mask.idxmax()
        dists, idxs = self.model.kneighbors(self.X[idx], n_neighbors=num_recommendations+1)
        rec_idxs = idxs.flatten()[1:]
        results = []
        for ridx, dist in zip(rec_idxs, dists.flatten()[1:]):
            row = df.iloc[ridx]
            results.append({
                "title": row["Series_Title"],
                "genre": row["Genre"],
                "similarity": 1 - float(dist),
                "rating": float(row["IMDB_Rating"]),
                "overview": row["Overview"]
            })
        return results 