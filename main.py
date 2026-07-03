"""
Main orchestration script for the machine failure prediction system.
Runs the complete pipeline: data generation, training, evaluation, visualization.
"""

import os
import sys


def run_pipeline():
    """Execute the complete ML pipeline."""
    
    print("="*70)
    print(" " * 15 + "MACHINE FAILURE PREDICTION SYSTEM")
    print("="*70)
    
    # Step 1: Generate Data
    print("\n[Step 1/4] Generating synthetic machine data...")
    print("-" * 70)
    try:
        from data_generator import MachineDataGenerator, create_engineered_features
        
        generator = MachineDataGenerator(n_machines=50, n_samples_per_machine=1000)
        df = generator.generate()
        df = create_engineered_features(df)
        df.to_csv('machine_data.csv', index=False)
        
        print(f"✓ Data generated successfully")
        print(f"  - Dataset shape: {df.shape}")
        print(f"  - Failure samples: {df['failure'].sum()} ({df['failure'].sum()/len(df)*100:.2f}%)")
        print(f"  - Saved to: machine_data.csv")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        sys.exit(1)
    
    # Step 2: Train Models
    print("\n[Step 2/4] Training machine learning models...")
    print("-" * 70)
    try:
        from model_training import FailurePredictionModel
        
        predictor = FailurePredictionModel()
        X, y = predictor.load_data('machine_data.csv')
        predictor.split_data(X, y)
        predictor.train_models()
        results = predictor.evaluate_models()
        predictor.feature_importance(X.columns)
        predictor.save_models()
        
        print(f"✓ Models trained and evaluated successfully")
        
        # Show best model
        best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
        print(f"  - Best model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
        
    except Exception as e:
        print(f"✗ Error training models: {e}")
        sys.exit(1)
    
    # Step 3: Create Visualizations
    print("\n[Step 3/4] Generating visualizations...")
    print("-" * 70)
    try:
        from visualization import (
            plot_model_comparison,
            plot_roc_curves,
            plot_precision_recall_curves,
            plot_confusion_matrices,
            plot_sensor_data,
            plot_failure_patterns
        )
        
        plot_model_comparison(results)
        plot_roc_curves(results, predictor.y_test)
        plot_precision_recall_curves(results, predictor.y_test)
        plot_confusion_matrices(results, predictor.y_test)
        plot_sensor_data(df)
        plot_failure_patterns(df)
        
        print(f"✓ Visualizations created successfully")
        print(f"  - Saved to: plots/ directory")
        
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        sys.exit(1)
    
    # Step 4: Demonstrate Inference
    print("\n[Step 4/4] Demonstrating inference on new data...")
    print("-" * 70)
    try:
        from predictor import demonstrate_prediction
        
        demonstrate_prediction()
        print(f"✓ Inference demonstration completed")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  ✓ machine_data.csv - Synthetic training data")
    print("  ✓ models/ - Trained ML models")
    print("  ✓ plots/ - Evaluation visualizations")
    print("\nNext steps:")
    print("  1. Review visualizations in plots/ directory")
    print("  2. Use predictor.py for inference on new data")
    print("  3. Check model_training.py for detailed metrics")
    print("="*70)


if __name__ == '__main__':
    run_pipeline()
