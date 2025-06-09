from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRecommendationModel(ABC):
    """Base class for all recommendation models."""
    
    def __init__(self, name: str, description: str, author: str):
        self.name = name
        self.description = description
        self.author = author
    
    @abstractmethod
    def recommend(self, movie_title: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate movie recommendations.
        
        Args:
            movie_title: The title of the movie to base recommendations on
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries containing movie recommendations
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the model and any necessary data."""
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author
        } 