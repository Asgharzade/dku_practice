from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """
    Train and evaluate a Random Forest Classifier.
    
    Parameters:
    -----------
    X_train, X_test : pandas DataFrame or numpy array
        Training and test feature sets
    y_train, y_test : pandas Series or numpy array
        Training and test target variables
    n_estimators : int, default=100
        Number of trees in the forest
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing model, predictions, and feature importance
    """
    # Initialize and train the model
    rfc = RandomForestClassifier(n_estimators=n_estimators, 
                                random_state=random_state,
                                n_jobs=-1)  # Use all available cores
    rfc.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rfc.predict(X_test)
    y_pred_proba = rfc.predict_proba(X_test)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rfc.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculate metrics
    metrics = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': feature_importance
    }
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.close()
    
    return {
        'model': rfc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics
    }

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot top N most important features.
    
    Parameters:
    -----------
    feature_importance : pandas DataFrame
        DataFrame containing feature names and their importance scores
    top_n : int, default=10
        Number of top features to plot
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(top_n),
                x='importance',
                y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.close()