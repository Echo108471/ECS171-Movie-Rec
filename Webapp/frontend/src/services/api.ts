import { ModelsResponse, MoviesResponse, RecommendationResponse } from '../types';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

export const api = {
    async getModels(): Promise<ModelsResponse> {
        const response = await fetch(`${API_BASE_URL}/models`);
        return response.json();
    },

    async getMovies(): Promise<MoviesResponse> {
        const response = await fetch(`${API_BASE_URL}/movies`);
        return response.json();
    },

    async getRecommendations(
        movie: string,
        model: string,
        numRecommendations: number
    ): Promise<RecommendationResponse> {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                movie,
                model,
                num_recommendations: numRecommendations,
            }),
        });
        return response.json();
    },
}; 