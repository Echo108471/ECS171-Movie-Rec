# Movie Recommendation System

**Authors:** Kyle Carbonell, Eugene Cho, Patrick Yao, and Brayan Torres Vega  
**Date:** June 8, 2025

## About This Project

This web app is the result of a research project exploring content-based movie recommendation using the IMDb Top 1000 dataset. Unlike typical recommenders that rely on user ratings, our system uses only movie features—such as plot summaries, genres, directors, cast, and ratings—to suggest similar films.

- **No user data required:** All recommendations are based solely on movie content.
- **Features used:** Plot (TF-IDF or SBERT), genres, director, cast, IMDb rating, and more.
- **Algorithm:** k-Nearest Neighbors (KNN) with cosine similarity in a high-dimensional feature space.
- **Research context:** Each model in the app represents a different experiment or ablation from our report, allowing users to compare the effect of different features and methods.

### How It Works

1. **Feature Engineering:**  
   - Plot summaries are vectorized using TF-IDF or SBERT.
   - Genres and top-4 cast members are multi-hot encoded.
   - Directors are one-hot encoded.
   - Ratings and votes are normalized.
2. **Similarity Search:**  
   - For any selected movie, the app finds the most similar titles using KNN and cosine similarity.
3. **Model Variants:**  
   - The dropdown lets you choose between models using different feature sets and methods, each mapped to a team member's research.
4. **Evaluation:**  
   - Our main metric is "genre-overlap@10": on average, 65% of the top-10 recommendations share at least one genre with the query movie.

### Why This Matters

Our results show that even without user histories, a well-engineered content-based approach can generate highly relevant movie suggestions. The app lets you explore how different features (plot, genre, director, etc.) affect recommendations, and demonstrates the strengths and limitations of content-based systems.

---

## Project Structure

```
MovieModel/
├── backend/           # Flask API (Poetry, SBERT, CSV)
├── frontend/          # React + Tailwind frontend
├── .gitignore
├── README.md
```

---

## Backend Setup (Flask + Poetry)

1. **Install Poetry:**
   https://python-poetry.org/docs/#installation

2. **Install dependencies:**
   ```bash
   cd backend
   poetry install
   ```

3. **Add .env (optional):**
   Create `backend/.env` for secrets/config (see `.env.example`).

4. **Run the backend:**
   ```bash
   poetry run flask run
   ```
   The API will be at `http://127.0.0.1:5000/api`.

---

## Frontend Setup (React + Tailwind)

1. **Install Node.js & npm:**
   https://nodejs.org/

2. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

3. **Add .env (optional):**
   Create `frontend/.env` with:
   ```env
   REACT_APP_API_URL=http://127.0.0.1:5000/api
   ```

4. **Run the frontend:**
   ```bash
   npm start
   ```
   The app will be at `http://localhost:3000`.

---

## Environment Files

- **backend/.env.example**
  ```env
  FLASK_ENV=development
  SECRET_KEY=your-secret-key
  ```
- **frontend/.env.example**
  ```env
  REACT_APP_API_URL=http://127.0.0.1:5000/api
  ```

---

## What to Commit
- All source code (backend/app, frontend/src, configs)
- `imdb_top_1000.csv` (in backend)
- `pyproject.toml`, `package.json`, configs
- `.gitignore`, `README.md`, `.env.example`

**Do NOT commit:**
- `.env` files with secrets
- `node_modules/`, `__pycache__/`, `.venv/`, etc.

---

## Credits
- Built by Brayan Torres and team.

---

## Model Variants in the Web App

Each model in the dropdown menu represents a different research experiment or ablation from our report. You can compare their recommendations and see how different features affect the results.

| Model Name (Dropdown)                | Author    | Methodology & Features Used                                                                 |
|--------------------------------------|-----------|--------------------------------------------------------------------------------------------|
| Overview + Genre (TF-IDF)            | Patrick   | TF-IDF on plot overview + one-hot encoded genres                                            |
| All Features Hybrid (TF-IDF, etc.)   | Kyle      | TF-IDF on plot, genres, director, cast, ratings (combined feature vector, KNN hybrid)       |
| Overview + Director (SBERT+TF-IDF)   | Eugene    | SBERT embeddings for plot + TF-IDF+SVD for director                                         |
| Overview + Genre (SBERT)             | Brayan    | SBERT embeddings for plot + genre filtering                                                 |

---

## Web App Features

- **Modern UI:** Built with React and Tailwind CSS for a clean, responsive experience.
- **Model Selection:** Use the dropdown to switch between different recommendation models and see how each performs.
- **Movie Selection:** Choose any movie from the IMDb Top 1000 as your seed for recommendations.
- **Horizontal Carousel:** Recommendations are displayed in a scrollable carousel for easy browsing.
- **Context Box:** The top of the app explains the project, research context, and team contributions.
- **Model Descriptions:** When you select a model, a short description appears to explain its methodology and author.
- **Consistent Results:** All models use KNN with cosine similarity, so you can compare their outputs fairly.
- **No User Data Needed:** All recommendations are content-based, so your privacy is protected.

--- 