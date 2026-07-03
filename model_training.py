"""
Train machine failure prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
import joblib
import warnings

warnings.filterwarnings('ignore')


class FailurePredictionModel:
    """Train and evaluate machine failure prediction models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, filepath):
        """Load and prepare data."""
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        # Drop non-feature columns
        X = df.drop(['machine_id', 'timestamp', 'failure'], axis=1)
        y = df['failure']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Failure samples: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets."""
        print(f"\nSplitting data ({test_size*100:.0f}% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train_models(self):
        """Train multiple models."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=self.random_state
        )
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['No Failure', 'Failure']))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix:")
            print(f"  True Negatives: {cm[0,0]}")
            print(f"  False Positives: {cm[0,1]}")
            print(f"  False Negatives: {cm[1,0]}")
            print(f"  True Positives: {cm[1,1]}")
            
            results[name] = {
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return results
    
    def feature_importance(self, feature_names):
        """Display feature importance for tree-based models."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        for name in ['Random Forest', 'Gradient Boosting']:
            if name in self.models:
                model = self.models[name]
                importances = model.feature_importances_
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:10]
                
                print(f"\nTop 10 Features - {name}:")
                print("-" * 40)
                for i, idx in enumerate(indices, 1):
                    print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    def save_models(self, directory='models'):
        """Save trained models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f"{directory}/{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save scaler
        scaler_path = f"{directory}/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")


def main():
    # Generate or load data
    try:
        print("Attempting to load existing data...")
        from data_generator import create_engineered_features
        df = pd.read_csv('machine_data.csv')
    except:
        print("Generating new data...")
        from data_generator import MachineDataGenerator, create_engineered_features
        generator = MachineDataGenerator()
        df = generator.generate()
        df = create_engineered_features(df)
        df.to_csv('machine_data.csv', index=False)
    
    # Load and prepare
    predictor = FailurePredictionModel()
    X, y = predictor.load_data('machine_data.csv')
    predictor.split_data(X, y)
    
    # Train models
    predictor.train_models()
    
    # Evaluate
    results = predictor.evaluate_models()
    
    # Feature importance
    predictor.feature_importance(X.columns)
    
    # Save models
    predictor.save_models()
    
    return predictor, results


if __name__ == '__main__':
    predictor, results = main()
