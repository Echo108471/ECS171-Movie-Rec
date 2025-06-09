import React, { useState, useEffect } from 'react';
import { ModelInfo, Movie } from './types';
import { api } from './services/api';

const MODEL_LABELS: Record<string, string> = {
    kyle: 'All Features (TF-IDF + Genre + Director + Cast + Ratings)',
    eugene: 'Overview + Director (SBERT + TFIDF)',
    sbert: 'Overview + Genre (SBERT)',
    patrick: 'Overview + Genre (TF-IDF)',
};

const MODEL_DESCRIPTIONS: Record<string, string> = {
    kyle: 'Combines plot (TF-IDF), genres, director, cast, and ratings into a single feature vector. Uses KNN with cosine similarity. This is the main model described in our research.',
    eugene: 'Uses SBERT embeddings for plot and TF-IDF+SVD for director. Explores the effect of deep semantic and director features.',
    sbert: 'Uses SBERT embeddings for plot and filters by genre overlap. Demonstrates the effect of using only plot and genre.',
    patrick: 'Uses TF-IDF on plot and one-hot encoded genres. Shows the effectiveness of traditional NLP techniques without deep learning.',
};

function App() {
    const [models, setModels] = useState<{ [key: string]: ModelInfo }>({});
    const [movies, setMovies] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [selectedMovie, setSelectedMovie] = useState<string>('');
    const [numRecommendations, setNumRecommendations] = useState<number>(5);
    const [recommendations, setRecommendations] = useState<Movie[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    useEffect(() => {
        const loadData = async () => {
            try {
                const [modelsData, moviesData] = await Promise.all([
                    api.getModels(),
                    api.getMovies(),
                ]);
                setModels(modelsData);
                setMovies(moviesData.movies);
                if (Object.keys(modelsData).length > 0) {
                    setSelectedModel(Object.keys(modelsData)[0]);
                }
            } catch (err) {
                setError('Failed to load initial data');
                console.error(err);
            }
        };
        loadData();
    }, []);

    const handleGetRecommendations = async () => {
        if (!selectedMovie || !selectedModel) return;
        setLoading(true);
        setError('');
        try {
            const response = await api.getRecommendations(
                selectedMovie,
                selectedModel,
                numRecommendations
            );
            setRecommendations(response.recommendations);
        } catch (err) {
            setError('Failed to get recommendations');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative min-h-screen flex flex-col items-center justify-center overflow-x-hidden">
            {/* Animated gradient background */}
            <div className="fixed inset-0 -z-10 animate-gradient bg-gradient-to-br from-[#23284a] via-[#2d325a] to-[#6c47c6] bg-[length:400%_400%]" style={{animation: 'gradientBG 12s ease-in-out infinite'}} />
            <style>{`
                @keyframes gradientBG {
                    0%, 100% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                }
            `}</style>

            {/* Title */}
            <h1 className="text-5xl md:text-6xl font-extrabold text-white drop-shadow-lg mb-6 mt-10 text-center tracking-tight">Movie Recommender</h1>

            {/* About This Project */}
            <section className="w-full max-w-2xl mx-auto mb-8 bg-white/10 backdrop-blur-xl rounded-2xl shadow-lg p-6 border border-white/10 flex flex-col items-center">
                <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                    <svg className="h-6 w-6 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A4 4 0 1116 12.25V12a4 4 0 01-1.248-7.832z" /></svg>
                    About This Project
                </h2>
                <p className="text-white/80 mb-2 text-center text-base">
                    <b>Movie Recommendation System</b> is a research project by Kyle Carbonell, Eugene Cho, Patrick Yao, and Brayan Torres Vega (2025). We use the IMDb Top 1000 dataset and explore different content-based recommendation models using only movie features (no user data).
                </p>
                <ul className="text-white/70 text-sm list-disc pl-5 mb-2 self-start">
                    <li>All models use <b>KNN with cosine similarity</b> to find similar movies.</li>
                    <li>Features include: plot (overview), genres, director, cast, and ratings.</li>
                    <li>Each model in the dropdown represents a different research experiment or feature combination from our report.</li>
                </ul>
                <div className="text-white/60 text-xs text-center">
                    <b>Research context:</b> Our main model combines all features (TF-IDF, genre, director, cast, ratings). We also include models using only plot+genre or plot+director, so you can explore how different features affect recommendations.
                </div>
            </section>
            {/* Two-column layout */}
            <div className="flex flex-row w-full max-w-6xl mx-auto mt-6 gap-8">
                {/* Left: Controls */}
                <div className="w-full md:w-1/2 p-6 bg-white/10 rounded-xl shadow-lg flex flex-col justify-start text-white">
                    <label htmlFor="model" className="block font-semibold mb-2 text-white/90">Select Model:</label>
                    <select
                        id="model"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full p-3 border border-white/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-white/20 text-white/90 placeholder:text-white/50 shadow-sm mb-2"
                    >
                        {Object.entries(models).map(([key, model]) => (
                            <option key={key} value={key} className="text-black">
                                {MODEL_LABELS[key] || model.name} by {model.author}
                            </option>
                        ))}
                    </select>
                    {selectedModel && (
                        <p className="text-white/70 text-sm mt-2 italic">
                            {MODEL_DESCRIPTIONS[selectedModel] || models[selectedModel].description}
                        </p>
                    )}

                    <label htmlFor="movie" className="block font-semibold mb-2 text-white/90 mt-6">Select Movie:</label>
                    <select
                        id="movie"
                        value={selectedMovie}
                        onChange={(e) => setSelectedMovie(e.target.value)}
                        className="w-full p-3 border border-white/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-white/20 text-white/90 placeholder:text-white/50 shadow-sm mb-2"
                    >
                        <option value="">Choose a movie...</option>
                        {movies.map((movie) => (
                            <option key={movie} value={movie} className="text-black">
                                {movie}
                            </option>
                        ))}
                    </select>

                    <label htmlFor="numRecommendations" className="block font-semibold mb-2 text-white/90 mt-6">
                        Number of Recommendations: <span className="font-bold text-indigo-200">{numRecommendations}</span>
                    </label>
                    <input
                        type="range"
                        id="numRecommendations"
                        min="1"
                        max="10"
                        value={numRecommendations}
                        onChange={(e) => setNumRecommendations(parseInt(e.target.value))}
                        className="w-full accent-indigo-400 mb-4"
                    />

                    <button
                        className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-bold rounded-2xl shadow-xl hover:from-indigo-600 hover:to-purple-600 transition disabled:bg-gray-300 disabled:cursor-not-allowed text-lg tracking-wide mt-2 ring-1 ring-indigo-400/20 hover:ring-indigo-400/40"
                        onClick={handleGetRecommendations}
                        disabled={!selectedMovie || !selectedModel || loading}
                    >
                        {loading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path></svg>
                                Getting Recommendations...
                            </span>
                        ) : 'Get Recommendations'}
                    </button>

                    {error && <div className="bg-red-400/20 text-red-200 p-3 rounded text-center font-semibold shadow mt-4">{error}</div>}
                </div>
                {/* Right: Recommendations */}
                <div className="w-full md:w-1/2 p-6 bg-white/10 rounded-xl shadow-lg flex flex-col justify-start text-white min-h-[400px]">
                    {recommendations.length > 0 && (
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-6 text-center flex items-center justify-center gap-2">
                                <svg className="h-7 w-7 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A4 4 0 1116 12.25V12a4 4 0 01-1.248-7.832z" /></svg>
                                Recommended Movies
                            </h2>
                            <div className="flex gap-6 overflow-x-auto pb-4 scroll-smooth scroll-px-4 snap-x snap-mandatory">
                                {recommendations.map((movie, index) => (
                                    <div
                                        key={index}
                                        className="min-w-[320px] max-w-xs bg-white/10 backdrop-blur-xl rounded-2xl shadow-xl p-6 border border-white/20 hover:scale-[1.025] hover:shadow-2xl transition-transform duration-200 relative overflow-hidden snap-center"
                                    >
                                        <div className="absolute right-4 top-4">
                                            <span className="inline-block bg-indigo-500/80 text-white text-xs font-bold px-3 py-1 rounded-full shadow">#{index + 1}</span>
                                        </div>
                                        <h3 className="text-xl font-bold text-indigo-200 mb-1 flex items-center gap-2">
                                            <span className="inline-block">🎬</span> {movie.title}
                                        </h3>
                                        <p className="text-white/60 text-sm mb-2 italic">{movie.genre}</p>
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-yellow-300 font-semibold flex items-center gap-1">
                                                <svg className="h-5 w-5 inline-block" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.049 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z" /></svg>
                                                {movie.rating}
                                            </span>
                                            <span className="text-green-300 font-semibold flex items-center gap-1">
                                                <svg className="h-5 w-5 inline-block" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4" /></svg>
                                                Similarity: {(movie.similarity * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <p className="text-white/90 text-base mt-2 leading-relaxed line-clamp-5">{movie.overview}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <footer className="mt-16 mb-4 text-center text-white/40 text-sm select-none">
                &copy; {new Date().getFullYear()} Movie Recommender &mdash; Built by Brayan Torres
            </footer>
        </div>
    );
}

export default App; 