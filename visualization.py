"""
Visualize machine failure predictions and model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_model_comparison(results, save_path='plots/model_comparison.png'):
    """Plot comparison of model metrics."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(results.keys())
    f1_scores = [results[m]['f1'] for m in models]
    roc_aucs = [results[m]['roc_auc'] for m in models]
    
    # F1 Scores
    axes[0].bar(models, f1_scores, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # ROC-AUC Scores
    axes[1].bar(models, roc_aucs, color='coral', alpha=0.8)
    axes[1].set_ylabel('ROC-AUC Score', fontsize=12)
    axes[1].set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison to {save_path}")
    plt.close()


def plot_roc_curves(results, y_test, save_path='plots/roc_curves.png'):
    """Plot ROC curves for all models."""
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results.items():
        y_pred_proba = metrics['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = metrics['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves to {save_path}")
    plt.close()


def plot_precision_recall_curves(results, y_test, save_path='plots/precision_recall.png'):
    """Plot precision-recall curves for all models."""
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results.items():
        y_pred_proba = metrics['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=model_name, linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved precision-recall curves to {save_path}")
    plt.close()


def plot_confusion_matrices(results, y_test, save_path='plots/confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    os.makedirs('plots', exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        y_pred = metrics['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True, annot_kws={'fontsize': 12})
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_xticklabels(['No Failure', 'Failure'])
        axes[idx].set_yticklabels(['No Failure', 'Failure'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_sensor_data(df, n_machines=5, save_path='plots/sensor_data.png'):
    """Plot sample sensor data from machines."""
    os.makedirs('plots', exist_ok=True)
    
    sensor_cols = ['temperature', 'vibration', 'pressure', 'rotation_speed']
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(14, 10))
    
    # Select first n_machines
    sample_df = df[df['machine_id'] <= n_machines].copy()
    
    for idx, col in enumerate(sensor_cols):
        for machine_id in range(1, n_machines + 1):
            machine_data = sample_df[sample_df['machine_id'] == machine_id][col].values
            axes[idx].plot(machine_data, alpha=0.6, label=f'Machine {machine_id}')
        
        axes[idx].set_ylabel(col.capitalize(), fontsize=11)
        axes[idx].set_xlabel('Sample Index', fontsize=11)
        axes[idx].set_title(f'{col.capitalize()} Over Time', fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensor data visualization to {save_path}")
    plt.close()


def plot_failure_patterns(df, save_path='plots/failure_patterns.png'):
    """Plot patterns in failure data."""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Failure distribution
    failure_counts = df['failure'].value_counts()
    axes[0, 0].bar(['No Failure', 'Failure'], failure_counts.values, 
                   color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Failure Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=11)
    
    # Temperature distribution
    no_fail = df[df['failure'] == 0]['temperature']
    fail = df[df['failure'] == 1]['temperature']
    axes[0, 1].hist(no_fail, alpha=0.6, label='No Failure', bins=30, color='green')
    axes[0, 1].hist(fail, alpha=0.6, label='Failure', bins=30, color='red')
    axes[0, 1].set_xlabel('Temperature (°C)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Temperature Distribution by Failure Status', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    
    # Vibration distribution
    no_fail = df[df['failure'] == 0]['vibration']
    fail = df[df['failure'] == 1]['vibration']
    axes[1, 0].hist(no_fail, alpha=0.6, label='No Failure', bins=30, color='green')
    axes[1, 0].hist(fail, alpha=0.6, label='Failure', bins=30, color='red')
    axes[1, 0].set_xlabel('Vibration (mm/s)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Vibration Distribution by Failure Status', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    
    # Pressure distribution
    no_fail = df[df['failure'] == 0]['pressure']
    fail = df[df['failure'] == 1]['pressure']
    axes[1, 1].hist(no_fail, alpha=0.6, label='No Failure', bins=30, color='green')
    axes[1, 1].hist(fail, alpha=0.6, label='Failure', bins=30, color='red')
    axes[1, 1].set_xlabel('Pressure (PSI)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Pressure Distribution by Failure Status', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved failure patterns to {save_path}")
    plt.close()


if __name__ == '__main__':
    import pandas as pd
    from model_training import main
    
    # Load data
    df = pd.read_csv('machine_data.csv')
    
    # Get predictions
    predictor, results = main()
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Plot visualizations
    plot_model_comparison(results)
    plot_roc_curves(results, predictor.y_test)
    plot_precision_recall_curves(results, predictor.y_test)
    plot_confusion_matrices(results, predictor.y_test)
    plot_sensor_data(df)
    plot_failure_patterns(df)
    
    print("\nAll visualizations saved to plots/ directory")
