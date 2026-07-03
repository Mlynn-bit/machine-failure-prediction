"""
Inference module for predicting machine failures on new data.
"""

import numpy as np
import pandas as pd
import joblib
import os


class FailurePredictor:
    """Make predictions on new machine sensor data."""
    
    def __init__(self, model_name='gradient_boosting', models_dir='models'):
        """
        Initialize the predictor.
        
        Args:
            model_name: Name of the model to use (without .pkl)
            models_dir: Directory containing saved models
        """
        self.models_dir = models_dir
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and scaler."""
        model_path = f"{self.models_dir}/{self.model_name}.pkl"
        scaler_path = f"{self.models_dir}/scaler.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Loaded model: {self.model_name}")
    
    def predict_single(self, sensor_data):
        """
        Predict failure for a single sample.
        
        Args:
            sensor_data: Dict with sensor readings
            
        Returns:
            Tuple of (prediction, probability)
        """
        # Create dataframe from single sample
        df = pd.DataFrame([sensor_data])
        
        # Scale
        X_scaled = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return prediction, probability
    
    def predict_batch(self, df):
        """
        Predict failures for multiple samples.
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with predictions and probabilities
        """
        # Scale
        X_scaled = self.scaler.transform(df)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create results dataframe
        results = df.copy()
        results['predicted_failure'] = predictions
        results['no_failure_prob'] = probabilities[:, 0]
        results['failure_prob'] = probabilities[:, 1]
        results['risk_level'] = results['failure_prob'].apply(self._risk_level)
        
        return results
    
    @staticmethod
    def _risk_level(probability):
        """Classify risk level based on failure probability."""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.7:
            return 'Medium'
        else:
            return 'High'
    
    def get_alert_threshold(self, threshold=0.5):
        """
        Get predictions with confidence above threshold.
        
        Args:
            threshold: Probability threshold for alerts
            
        Returns:
            DataFrame with high-confidence predictions
        """
        # This should be called on results from predict_batch
        pass


def demonstrate_prediction():
    """Demonstrate making predictions."""
    print("="*60)
    print("MACHINE FAILURE PREDICTION - INFERENCE DEMO")
    print("="*60)
    
    try:
        predictor = FailurePredictor(model_name='gradient_boosting')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run model_training.py first to train and save models.")
        return
    
    # Example 1: Predict single sample with normal readings
    print("\n1. Predicting single sample - NORMAL OPERATION:")
    normal_sample = {
        'temperature': 72.5,
        'vibration': 2.3,
        'pressure': 102.1,
        'rotation_speed': 1510.0,
        'temperature_rolling_mean': 71.8,
        'temperature_rolling_std': 2.1,
        'temperature_rate_change': 0.5,
        'vibration_rolling_mean': 2.4,
        'vibration_rolling_std': 0.3,
        'vibration_rate_change': 0.1,
        'pressure_rolling_mean': 101.5,
        'pressure_rolling_std': 1.2,
        'pressure_rate_change': 0.3,
        'rotation_speed_rolling_mean': 1505.0,
        'rotation_speed_rolling_std': 20.0,
        'rotation_speed_rate_change': 5.0,
        'temp_vibration_interaction': 166.75,
        'pressure_rotation_interaction': 153819.1
    }
    
    pred, prob = predictor.predict_single(normal_sample)
    print(f"Prediction: {'FAILURE' if pred == 1 else 'NO FAILURE'}")
    print(f"Failure Probability: {prob[1]:.4f}")
    print(f"No Failure Probability: {prob[0]:.4f}")
    
    # Example 2: Predict single sample with degraded readings
    print("\n2. Predicting single sample - DEGRADED OPERATION:")
    degraded_sample = {
        'temperature': 95.0,  # High
        'vibration': 6.2,     # High
        'pressure': 85.5,     # Low
        'rotation_speed': 1480.0,
        'temperature_rolling_mean': 90.5,
        'temperature_rolling_std': 5.2,
        'temperature_rate_change': 2.5,
        'vibration_rolling_mean': 5.8,
        'vibration_rolling_std': 1.1,
        'vibration_rate_change': 0.8,
        'pressure_rolling_mean': 87.0,
        'pressure_rolling_std': 3.5,
        'pressure_rate_change': -1.2,
        'rotation_speed_rolling_mean': 1485.0,
        'rotation_speed_rolling_std': 25.0,
        'rotation_speed_rate_change': -8.0,
        'temp_vibration_interaction': 589.0,
        'pressure_rotation_interaction': 126474.0
    }
    
    pred, prob = predictor.predict_single(degraded_sample)
    print(f"Prediction: {'FAILURE' if pred == 1 else 'NO FAILURE'}")
    print(f"Failure Probability: {prob[1]:.4f}")
    print(f"No Failure Probability: {prob[0]:.4f}")
    
    # Example 3: Batch prediction
    print("\n3. Batch prediction on test data:")
    try:
        df = pd.read_csv('machine_data.csv')
        
        # Select a subset (last 50 rows)
        test_df = df.tail(50).copy()
        feature_cols = [col for col in df.columns 
                       if col not in ['machine_id', 'timestamp', 'failure']]
        X_test = test_df[feature_cols]
        
        results = predictor.predict_batch(X_test)
        
        # Show predictions with high risk
        high_risk = results[results['risk_level'] == 'High']
        print(f"\nTotal samples: {len(results)}")
        print(f"Predicted failures: {results['predicted_failure'].sum()}")
        print(f"High risk alerts: {len(high_risk)}")
        
        print("\nSample results (first 5):")
        display_cols = ['predicted_failure', 'no_failure_prob', 'failure_prob', 'risk_level']
        print(results[display_cols].head())
        
    except FileNotFoundError:
        print("machine_data.csv not found. Generate it first using data_generator.py")


if __name__ == '__main__':
    demonstrate_prediction()
