from flask import Blueprint, jsonify, request
from ..models.sbert_model import SBERTModel
from ..models.kyle_model import KyleModel
from ..models.eugene_model import EugeneModel
from ..models.patrick_model import PatrickModel
from typing import Dict, List

api = Blueprint('api', __name__)

# Initialize models
models: Dict[str, SBERTModel] = {
    "sbert": SBERTModel(),
    "kyle": KyleModel(),
    "eugene": EugeneModel(),
    "patrick": PatrickModel(),
}
models["sbert"].load_model()  # Eagerly load the SBERT model and CSV
models["kyle"].load_model()   # Eagerly load Kyle's model
models["eugene"].load_model() # Eagerly load Eugene's model
models["patrick"].load_model() # Eagerly load Patrick's model

@api.route('/models', methods=['GET'])
def get_models():
    """Get information about available models."""
    return jsonify({
        name: model.get_info()
        for name, model in models.items()
    })

@api.route('/movies', methods=['GET'])
def get_movies():
    """Get list of available movies."""
    # Use the SBERT model's dataset since it's already loaded
    movies = models["sbert"].df["Series_Title"].tolist()
    return jsonify({"movies": movies})

@api.route('/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations."""
    data = request.get_json()
    movie_title = data.get('movie')
    model_name = data.get('model', 'sbert')
    num_recommendations = data.get('num_recommendations', 5)
    
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    model = models[model_name]
    recommendations = model.recommend(movie_title, num_recommendations)
    
    return jsonify({
        "model": model.get_info(),
        "recommendations": recommendations
    }) 