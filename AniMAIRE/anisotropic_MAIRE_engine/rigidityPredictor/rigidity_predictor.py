"""
Rigidity Predictor Module

This module provides functionality for predicting cosmic ray cutoff rigidities
using machine learning models.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, List, Optional, Union, Tuple
try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files

class RigidityPredictor:
    """
    A class for predicting cosmic ray cutoff rigidities.
    
    This class uses a trained machine learning model to predict cutoff rigidities
    (Ru, Rc, Rl) for given locations and conditions.
    
    Attributes:
        model: The trained machine learning model
        scaler: The feature scaler used during training
        feature_names: List of feature names used in the model
    """
    
    def __init__(self, model=None, scaler=None, feature_names=None):
        """Initialize the predictor with a trained model"""
        self.model = model
        # Get scaler and feature_names from model if available
        if model is not None and hasattr(model, 'scaler'):
            self.scaler = model.scaler
        else:
            self.scaler = scaler
            
        if model is not None and hasattr(model, 'feature_names'):
            self.feature_names = model.feature_names
        else:
            self.feature_names = feature_names
        
    def _create_world_grid(self, resolution: int = 5) -> np.ndarray:
        """Create a grid of lat/lon points"""
        lats = np.arange(-90, 90 + resolution, resolution)
        lons = np.arange(0, 360, resolution)
        return np.array([(lat, lon) for lat in lats for lon in lons])
    
    def _prepare_features(self, latitude: float, longitude: float, kp: float, 
                         datetime_obj: Optional[datetime] = None, 
                         altitude: float = 100) -> np.ndarray:
        """Prepare features for a single prediction point"""
        if datetime_obj is None:
            datetime_obj = datetime.now()
        
        # Calculate cyclical features
        hour = datetime_obj.hour
        hour_sin = np.sin(2 * np.pi * hour/24)
        hour_cos = np.cos(2 * np.pi * hour/24)
        
        lat_sin = np.sin(2 * np.pi * latitude/360)
        lat_cos = np.cos(2 * np.pi * latitude/360)
        lon_sin = np.sin(2 * np.pi * longitude/360)
        lon_cos = np.cos(2 * np.pi * longitude/360)
        
        # Year as absolute value instead of periodic
        year = datetime_obj.year
        
        # Interaction features
        kp_lat = kp * latitude
        kp_lon = kp * longitude
        
        # Calculate McIlwain L parameter (simplified approximation)
        # This is a placeholder - in a real implementation you would use a proper model
        # For now, we'll use a simple approximation based on latitude
        mcilwain_l = 1.0 / (np.cos(np.radians(latitude)) ** 2) if abs(latitude) < 85 else 99.99
        
        # Create feature array matching the training features
        feature_dict = {
            'kp': kp,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'lat_sin': lat_sin,
            'lat_cos': lat_cos,
            'lon_sin': lon_sin,
            'lon_cos': lon_cos,
            'kp_lat': kp_lat,
            'kp_lon': kp_lon,
            'year': year,  # Using absolute year value
            'altitude': altitude,
            'McIlwain_L': mcilwain_l
        }
        
        # Ensure we use the same features in the same order as during training
        features = np.array([feature_dict.get(feature, 0) for feature in self.feature_names])
        
        return features.reshape(1, -1)
    
    def predict_single_point(self, latitude: float, longitude: float, kp: float, 
                           datetime_obj: Optional[datetime] = None, 
                           altitude: float = 100) -> Dict[str, float]:
        """Predict rigidity for a single point"""
        features = self._prepare_features(latitude, longitude, kp, datetime_obj, altitude)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        
        # Model predicts Ru directly, Rc as fraction of Ru, and Rl as fraction of Rc
        ru = prediction[0][0]
        rc_fraction = prediction[0][1]
        rl_fraction = prediction[0][2]
        
        # Convert fractions to actual values
        rc = ru * rc_fraction
        rl = rc * rl_fraction
        
        return {
            'Ru': ru,
            'Rc': rc,
            'Rl': rl
        }
    
    def predict_world_map(self, kp: float, datetime_obj: Optional[datetime] = None, 
                         resolution: int = 5, altitude: float = 100) -> pd.DataFrame:
        """Predict rigidity for entire world map"""
        grid_points = self._create_world_grid(resolution)
        predictions = []
        
        # Create a batch of features for all grid points
        features_batch = []
        for lat, lon in grid_points:
            features = self._prepare_features(lat, lon, kp, datetime_obj, altitude)
            features_batch.append(features[0])
        
        # Scale all features at once
        features_batch = np.array(features_batch)
        features_scaled = self.scaler.transform(features_batch)
        
        # Predict all points at once
        batch_predictions = self.model.predict(features_scaled)
        
        # Process predictions
        for i, (lat, lon) in enumerate(grid_points):
            pred = batch_predictions[i]
            ru = pred[0]
            rc = ru * pred[1]
            rl = rc * pred[2]
            
            predictions.append({
                'latitude': lat,
                'longitude': lon,
                'Ru': ru,
                'Rc': rc,
                'Rl': rl
            })
        
        return pd.DataFrame(predictions)
    
    def plot_world_map(self, predictions_df: pd.DataFrame, value_column: str = 'Rc', 
                      title: Optional[str] = None, cmap: str = 'viridis', 
                      resolution: int = 5) -> plt.Figure:
        """Plot predictions on a world map"""
        plt.figure(figsize=(15, 10))
        
        # Reshape data for plotting
        lats = sorted(predictions_df['latitude'].unique())
        lons = sorted(predictions_df['longitude'].unique())
        values = predictions_df.pivot(
            index='latitude', 
            columns='longitude', 
            values=value_column
        ).values
        
        # Create mesh grid
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        # Plot
        plt.contourf(lon_mesh, lat_mesh, values, levels=50, cmap=cmap)
        plt.colorbar(label=f'{value_column} (GV)')
        
        # Customize plot
        plt.title(title or f'World Map of {value_column}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple points from a DataFrame"""
        features_list = []
        
        for _, row in df.iterrows():
            features = self._prepare_features(
                row['latitude'], 
                row['longitude'], 
                row['kp'],
                row.get('datetime', None),
                row.get('altitude', 100)
            )
            features_list.append(features[0])
        
        features_array = np.array(features_list)
        features_scaled = self.scaler.transform(features_array)
        predictions = self.model.predict(features_scaled)
        
        results_df = df.copy()
        
        # Convert fractional predictions to actual values
        results_df['Ru'] = predictions[:, 0]
        results_df['Rc'] = results_df['Ru'] * predictions[:, 1]
        results_df['Rl'] = results_df['Rc'] * predictions[:, 2]
        
        return results_df
    
    def save(self, filepath: str) -> None:
        """Save the predictor model and associated data to a file"""
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
            
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the model and associated data
        joblib.dump(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'RigidityPredictor':
        """
        Load a predictor model from a file or use the default model.
        
        Args:
            filepath: Path to the model file. If None, loads the default model
                     from the package's data directory.
        
        Returns:
            RigidityPredictor instance
        """
        if filepath is None:
            # Load the default model from package data
            try:
                package_files = files('AniMAIRE.anisotropic_MAIRE_engine.rigidityPredictor')
                data_path = package_files / 'data' / 'cutoff_rigidity_predictor.pkl'
                filepath = str(data_path)
            except Exception as e:
                raise FileNotFoundError(
                    "Default model file not found in package data directory. "
                    "Please provide a specific filepath or ensure the package "
                    "is installed correctly."
                ) from e
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load the saved dictionary
        save_dict = joblib.load(filepath)
        
        # Create a new instance
        predictor = cls(
            model=save_dict['model'],
            scaler=save_dict['scaler'],
            feature_names=save_dict['feature_names']
        )
        
        print(f"Model loaded from {filepath}")
        return predictor