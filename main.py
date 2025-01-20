import pandas as pd
import numpy as np
from ML.rfc import train_and_evaluate_rf, plot_feature_importance
from postprocessing import post_processing

train_dir = 'data/processed/census_income_learn.csv'
test_dir = 'data/processed/census_income_test.csv'

# Load the train/test data
print("Train data: ###############################")
X_train, y_train = post_processing(train_dir)
print(f'X_train shape: {X_train.shape}')

print("\nTest data: ###############################")
X_test, y_test = post_processing(test_dir)
print(f'X_test shape: {X_test.shape}')



