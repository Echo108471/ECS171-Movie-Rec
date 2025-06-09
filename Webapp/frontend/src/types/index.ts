export interface ModelInfo {
    name: string;
    description: string;
    author: string;
}

export interface Movie {
    title: string;
    genre: string;
    similarity: number;
    rating: number;
    overview: string;
}

export interface RecommendationResponse {
    model: ModelInfo;
    recommendations: Movie[];
}

export interface ModelsResponse {
    [key: string]: ModelInfo;
}

export interface MoviesResponse {
    movies: string[];
} 