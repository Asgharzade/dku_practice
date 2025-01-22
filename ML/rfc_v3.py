import logging
import pandas as pd
import numpy as np
from postprocessing import post_processing
from imblearn.under_sampling import OneSidedSelection
from collections import Counter
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import optuna
import pickle
import os
import datetime

# Configure logging
filename = os.path.basename(__file__).split('.')[0]
ml_dir = 'ML'


def rfc(ml_dir = 'ML', filename = filename):
    """
    Random Forest Classifier with One Sided Selection and Optuna Hyperparameter Tuning
    Args:
        ml_dir (str): Directory for machine learning outputs.
        filename (str): Name of the file for logging and saving outputs.
    Returns:
        None
    """
    train_dir = 'data/processed/census_income_learn.csv'
    test_dir = 'data/processed/census_income_test.csv'
    logging.basicConfig(filename=os.path.join(ml_dir, f'{filename}_report.log'), level=logging.INFO, format='%(message)s', filemode='w')
    # Load the train/test data
    logging.info(f'fitting model {filename} started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}')
    logging.info("\nTrain data: ###############################")
    X_train, y_train = post_processing(train_dir, get_dummies= True, over_18= True)
    X_train = X_train.astype(int)
    logging.info(f'\nX_train shape: {X_train.shape}')

    logging.info("\nTest data: ###############################")
    X_test, y_test = post_processing(test_dir, get_dummies= True, over_18= True)
    X_test = X_test.astype(int)
    logging.info(f'\nX_test shape: {X_test.shape}')

    # Perform OSS
    oss = OneSidedSelection(random_state=42)
    X_train_oss, y_train_oss = oss.fit_resample(X_train, y_train)

    # Print class distribution before and after OSS
    logging.info("\nOriginal class distribution:")
    logging.info(Counter(y_train))
    logging.info("\nResampled class distribution after OSS:")
    logging.info(Counter(y_train_oss))

    # Create dataset for LightGBM using OSS resampled data
    train_data = lgb.Dataset(X_train_oss, y_train_oss)
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)

    # Define the objective function for Optuna
    
    num_rounds = 100
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': 0
        }
        
        # Train model with current parameters
        model = lgb.train(
            params,
            train_data,
            num_rounds,
            valid_sets=[test_data],
            # early_stopping_rounds=20,
            # verbose_eval=False
        )
        
        # Get validation score
        y_pred = model.predict(X_test)
        return roc_auc_score(y_test, y_pred)

    # Create and run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': 0
    })

    logging.info("\nBest parameters found: %s", best_params)

    # Train final model with best parameters
    num_rounds = 100
    model = lgb.train(
        best_params,
        train_data,
        num_rounds,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid']
    )

    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.round(y_pred_proba)

    # Evaluate model
    logging.info("\nModel Evaluation:")
    logging.info("-----------------")
    logging.info(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"\nROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    logging.info("\nClassification Report:")
    logging.info(f'\n{classification_report(y_test, y_pred)}')

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    logging.info("\nTop 10 Most Important Features:")
    logging.info(f'\n{feature_importance.head(15)}')

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info('\nconfusion_matrix:')
    logging.info(cm)
    logging.info("--------------------")
    # Print confusion matrix values in a tabular format
    logging.info("\nConfusion Matrix Values:")
    logging.info("--------------------")
    logging.info(f"\nTrue Negatives (TN): {cm[0][0]}")
    logging.info(f"\nFalse Positives (FP): {cm[0][1]}")
    logging.info(f"\nFalse Negatives (FN): {cm[1][0]}")
    logging.info(f"\nTrue Positives (TP): {cm[1][1]}")

    # Calculate and print percentages
    total = cm.sum()
    logging.info("\nConfusion Matrix Percentages:")
    logging.info("\n-------------------------")
    logging.info(f"\nTrue Negatives (TN): {(cm[0][0]/total)*100:.2f}%")
    logging.info(f"\nFalse Positives (FP): {(cm[0][1]/total)*100:.2f}%")
    logging.info(f"\nFalse Negatives (FN): {(cm[1][0]/total)*100:.2f}%")
    logging.info(f"\nTrue Positives (TP): {(cm[1][1]/total)*100:.2f}%")

    # Plot and analyze feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(f'ML/{filename}feature_importance.png')
    # plt.show()

    # Analyze correlation between top features and predictions
    top_features = feature_importance['feature'][:5].tolist()
    correlations = pd.DataFrame()
    for feature in top_features:
        correlations[feature] = pd.Series({
            'correlation_with_prediction': np.corrcoef(X_test[feature], y_pred)[0,1],
            'mean_value_when_predicted_1': X_test[feature][y_pred == 1].mean(),
            'mean_value_when_predicted_0': X_test[feature][y_pred == 0].mean()
        })

    logging.info("\nFeature Correlation Analysis:")
    logging.info(f'\n{correlations}')
    logging.info("\n--------------------")

    # Model performance metrics
    logging.info("\nDetailed Model Performance Metrics:")
    logging.info(f"\nAccuracy: {np.mean(y_pred == y_test):.4f}")
    logging.info(f"\nTrue Positive Rate: {cm[1,1] / (cm[1,0] + cm[1,1]):.4f}")
    logging.info(f"\nTrue Negative Rate: {cm[0,0] / (cm[0,0] + cm[0,1]):.4f}")
    logging.info(f"\nPositive Predictive Value: {cm[1,1] / (cm[0,1] + cm[1,1]):.4f}")

    # Distribution of probabilities
    plt.figure(figsize=(10, 5))
    plt.hist(y_pred_proba, bins=50)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig(f'ML/{filename}_prediction_probabilities.png')

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_curve_vals = {'fpr' : fpr, 'tpr' : tpr}
    roc_curve_vals = pd.DataFrame(roc_curve_vals)
    logging.info(f'\n{roc_curve_vals}')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'ML/{filename}_roc_curve.png')

    # Additional model insights
    logging.info("\nModel Configuration:")
    logging.info("\n--------------------")
    logging.info(f"\nNumber of training rounds: {num_rounds}")
    logging.info("\nModel Parameters:")
    for key, value in best_params.items():
        logging.info(f"\n{key}: {value}")

    # Training data balance analysis
    logging.info("\nClass Distribution in Training Data:")
    logging.info(f"\nClass counts: \n{y_train.value_counts()}")
    logging.info(f"\nClass proportions: \n{y_train.value_counts(normalize=True)}")
    # Print model's training history
    logging.info("\nTraining History:")
    logging.info("\n----------------")
    evals_result = {}
    for metric, values in model.best_score.items():
        logging.info(f"\n{metric}: {values}")

    # Feature importance percentages
    total_importance = feature_importance['importance'].sum()
    feature_importance['importance_percentage'] = (feature_importance['importance'] / total_importance * 100).round(2)
    logging.info("\nFeature Importance Percentages (Top 10):")
    logging.info(f'\n{feature_importance[['feature', 'importance_percentage']].head(15)}')

    # Calculate prediction threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_metrics = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        threshold_metrics.append({'threshold': thresh, 'accuracy': acc})

    threshold_df = pd.DataFrame(threshold_metrics)
    logging.info("\nPrediction Threshold Analysis:")
    logging.info(f'\n{threshold_df}')

    # Plot threshold analysis
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df['threshold'], threshold_df['accuracy'], marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Prediction Threshold')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'ML/{filename}_threshold_analysis.png')

    # Model complexity analysis
    logging.info("\nModel Complexity Analysis:")
    logging.info(f"\nNumber of trees: {model.num_trees()}")
    logging.info(f"\nTotal number of leaves: {model.num_trees() * best_params['num_leaves']}")

    # Analyze prediction confidence
    confidence_bins = pd.cut(y_pred_proba, bins=15)
    confidence_analysis = pd.DataFrame({
        'prediction_confidence': confidence_bins,
        'actual': y_test
    }).groupby('prediction_confidence').agg({
        'actual': ['count', 'mean']
    })

    logging.info("\nPrediction Confidence Analysis:")
    logging.info(f'\n{confidence_analysis}')        




    # Save the model as a pickle file
    
    with open(f'{ml_dir}/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info("\nModel saved as best_model.pkl")