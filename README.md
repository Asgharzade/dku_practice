# A Classification Analysis of Demographic Factors Influencing Annual Earnings Using U.S. Census Data

As a data scientist with a strong foundation in statistical analysis and machine learning, I am excited to tackle the challenge presented by the United States Census Bureau's dataset. This project involves analyzing a sample of anonymized demographic and economic data from approximately 300,000 individuals to identify characteristics associated with annual incomes of more or less than $50,000. My approach will encompass a comprehensive data analysis and modeling pipeline, beginning with exploratory data analysis to uncover insights through numerical and graphical representations. I will meticulously prepare the data through cleaning, preprocessing, and feature engineering to enhance clarity and model performance. Following this, I will construct and evaluate multiple predictive models, selecting the best-performing one based on rigorous assessment criteria. The culmination of this project will be a concise summary of key findings, actionable recommendations, and suggestions for future improvements. I am committed to delivering not only robust analytical results but also an engaging presentation that effectively communicates insights and fosters collaboration, reflecting my dedication to customer support and proficiency in data science.

## Requirements
### System Requirements
Python 3.8 or higher installed on your system.

### Python Requirements
Preferably, install packages in a virtual environment.


### Run the Project
1. Clone the repository: 
```sh
git clone https://github.com/Asgharzade/dku_practice.git
```
2. To install the required Python packages:
```sh
pip install -r requirements.txt
```

3. run ```main.py```. This file runs the pre-processing and store the processed train/test datasets in the ```data/processed``` dir. 
```sh
python main.py
```

# Summary of Process

### Steps

1. Variable Mapping
2. Dropping Duplicates
3. Remapping
4. Pre-Processing (Under/Oversampling)
5. Modeling
6. Results and Discussion

## Variable Mapping

| Column Name                                 | Data Type | Required | Mapping | Considered?|
|---------------------------------------------|-----------|----------|---------|----------------|
| age                                         | int64     | y        |         |yes, only over 18    |
| class_of_worker                             | object    | y        | map     |yes, keep only 5 cats|
| detailed_industry_recode                    | object    | y        | map     |yes,high corr with class_of_worker|
| detailed_occupation_recode                  | object    | y        | map     |no,high corr with class_of_worker|
| education                                   | object    | y        | map     |yes|
| wage_per_hour                               | int64     | y        |         |yes|
| enroll_in_edu_inst_last_wk                  | object    | y        |         |yes|
| marital_stat                                | object    | y        | map     |yes|
| major_industry_code                         | object    | y        |         |yes|
| major_occupation_code                       | object    | y        |         |yes|
| race                                        | object    | y        |         |yes, but might be sensitive|
| hispanic_origin                             | object    | n        |         |not relative|
| sex                                         | object    | y        |         |yes, but might be sensitive|
| member_of_a_labor_union                     | object    | y        |         |yes, just keep 'Yes'|
| reason_for_unemployment                     | object    | y        |         |yes|
| full_or_part_time_employment_stat           | object    | y        | map     |yes|
| capital_gains                               | int64     | y        |         |yes|
| capital_losses                              | int64     | y        |         |yes|
| dividends_from_stocks                       | int64     | y        |         |yes|
| tax_filer_stat                              | object    | y        | map     |yes|
| region_of_previous_residence                | object    | y        |         |yes. just four cats|
| state_of_previous_residence                 | object    | n        |         |no, too many cats|
| detailed_household_and_family_stat          | object    | y        | map     |yes|
| detailed_household_summary_in_household     | object    | y        | map     |no, too many cats|
| instance_weight                             | float64   | n        |         |no, instructed to be ignored|
| migration_code_change_in_msa                | object    | y        | map     |no, high corr with move_reg|
| migration_code_change_in_reg                | object    | y        |         |yes, only two cats|
| migration_code_move_within_reg              | object    | n        |         |no, high corr with move_reg|
| live_in_this_house_1_year_ago               | object    | y        |         |yes|
| migration_prev_res_in_sunbelt               | object    | y        |         |yes|
| num_persons_worked_for_employer             | int64     | y        |         |yes|
| family_members_under_18                     | object    | n        |         |yes|
| country_of_birth_father                     | object    | y        |         |no, high corr with citizenship|
| country_of_birth_mother                     | object    | y        |         |no, high corr with citizenship|
| country_of_birth_self                       | object    | y        |         |no, high corr with citizenship|
| citizenship                                 | object    | y        |         |yes|
| own_business_or_self_employed               | object    | y        | map     |yes|
| fill_inc_questionnaire_for_veterans_admin   | object    | y        |         |no, using veterans_benefits instead|
| veterans_benefits                           | object    | y        |         |yes|
| weeks_worked_in_year                        | int64     | y        |         |yes|
| year                                        | int64     | y        |         |not related|
| target                                      | object    | T        |         |Target|



### Dropping Duplicates
There were 3229 duplicated records.. All belong to the -50k category

### Dropping people younger than 18
All individuals who are younger than 18 year-old are dropped


## Remapping
Remapping was done for perfrom feature engineering and grouping various.
The remapped features are as follows:
* detailed_industry_recode
* detailed_occupation_recode
* education
* marital_stat
* reason_for_unemployment
* full_or_part_time_employment_stat
* migration_prev_res_in_sunbelt
* migration_code_change_in_reg
* migration_code_change_in_msa
* detailed_household_and_family_stat
* migration_code_change_in_reg
* tax_filer_stat
* class_of_worker
* citizenship
* veterans_benefits
* own_business_or_self_employed
* migration_prev_res_in_sunbelt
* live_in_this_house_1_year_ago

## Imbalanced Data
Das data were inhenrently imbalanced, we should perfrom resampling.
our options are
* undersampling
* oversampling

## Under-Sampling
Popular undersampling methods in machine learning include:

1. Random Undersampling (RUS)
   - Randomly removes majority class samples
   - Simple but can lose important information

2. Tomek Links
   - Removes majority class samples near decision boundary
   - Helps clean overlapping class regions

3. Cluster Centroids
   - Uses clustering to create representative samples of majority class
   - Reduces dataset while preserving class distribution characteristics

4. Near Miss
   - Selects majority class samples closest to minority class instances
   - Variants (Near Miss-1, Near Miss-2, Near Miss-3) use different proximity strategies

5. One-Sided Selection (**probably the best option for the census data**)
   - Combines Tomek Links with Condensed Nearest Neighbor Rule
   - Removes borderline and noisy majority class samples

6. EasyEnsemble
   - Creates multiple subsets by randomly sampling majority class
   - Trains multiple classifiers on these balanced subsets

### One-Sided Selection (OSS)

Census data frequently exhibits inherent redundancy due to analogous demographic trends.
OSS effectively retains significant boundary cases while eliminating redundant majority samples.
It is particularly suitable for categorical variables that are prevalent in census data.
The two-step methodology (Tomek Links + CNN) aids in preserving crucial demographic patterns.
It is less likely to disrupt the natural distribution of various population segments.

The Neighborhood Cleaning Rule (NCR) ranks as the second-best option due to the following factors:

It is proficient in identifying and eliminating noise while maintaining structural integrity.
It performs well with a combination of numerical and categorical features.
It upholds local demographic patterns within the dataset.
It is less aggressive compared to random undersampling.

Random undersampling is deemed excessively destructive for census data, as it may result in the removal of entire demographic groups or critical patterns. Likewise, relying solely on Tomek Links may overly concentrate on boundary cases, potentially overlooking broader population trends.

The decision between OSS and NCR may be influenced by:

The specific imbalance ratio present in the census data.
The significance of boundary cases in the classification task at hand.
The computational resources at your disposal (OSS may require more computational power).
The particular demographic patterns that need to be preserved.

### OverSampling (another option)

1. SMOTE (Synthetic Minority Over-sampling Technique)
   - Generates synthetic samples for numerical features
   - Creates samples along line segments joining minority class instances

2. SMOTENC (**probably the best options for census data**)
   - Handles both numerical and categorical features
   - Generates synthetic samples while preserving categorical variable characteristics
   - Adapts SMOTE for mixed data types

3. ADASYN (Adaptive Synthetic)
   - Creates synthetic samples focusing on harder-to-learn minority instances
   - Weights sample generation based on learning difficulty

4. Borderline-SMOTE
   - Generates synthetic samples near decision boundary
   - More targeted than standard SMOTE

5. SMOTEENN
   - Combines SMOTE with Edited Nearest Neighbors
   - Removes overlapping and noisy samples after synthetic generation

6. K-Means SMOTE
   - Uses K-means clustering to guide synthetic sample generation
   - Improves diversity of synthetic samples

### Modeling
Since the dataset is already labeled, this is a basic binary classifiation problem. 
Here are some potential machine learning classification models:

Probabilistic Models:
- Logistic Regression
- Naive Bayes
- Bayesian Networks

Tree-Based Models:
- Decision Trees
- Random Forest (**This method was also selected for this project**)
- Gradient Boosting Trees
- XGBoost
- Light GBM

Support Vector Machines:
- Linear SVM
- Kernel SVM (RBF, Polynomial)

Neural Network Models:
- Multilayer Perceptron
- Convolutional Neural Networks
- Recurrent Neural Networks

Ensemble Methods:
- AdaBoost
- Stacking Classifiers
- Voting Classifiers

Distance-Based Models:
- K-Nearest Neighbors
- Support Vector Neighbors

Advanced/Specialized Models:
- LightGBM (**This method was selected for this project**)
- CatBoost
- XGBoost
- Support Vector Machines with Gaussian Kernels

Why LightGBM:
* Benefits:
    * Rapid training time on large datasets
    * Handles categorical features natively
    * Gradient-based one-side sampling reduces overfitting
    * High accuracy with complex feature interactions
    * Memory efficient with leaf-wise tree growth
    * Great performance on tabular data like census
* Drawbacks
    * Can overfit on small datasets.
    * Hyper-parameter tuning complexity
    * Less interpretable than linear models
    * Sensitive to noisy features

### Result and Summary

#### Train Test Data Information 
```sh
# Train data:
X_train_shape: (143468, 82)
# Test data:
X_test_shape: (72023, 82)
```
```
Original class distribution:
Counter({1: 131088, 0: 12380})

Resampled class distribution after OSS:
Counter({1: 128177, 0: 12380})
```
Best parameters found: 
```json
{
    "num_leaves": 98,
    "learning_rate": 0.0994,
    "feature_fraction": 0.5396,
    "bagging_fraction": 0.9906,
    "bagging_freq": 1,
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbose": 0
}
```

#### Model Evaluation:
```
-----------------
Accuracy: 0.9400
ROC AUC: 0.9333
Classification Report:

              precision    recall  f1-score   support

           0       0.72      0.49      0.58      6186
           1       0.95      0.98      0.97     65837

    accuracy                           0.94     72023
```

#### Metrics

| Metric          | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Macro Avg       | 0.84      | 0.74   | 0.78     | 72023   |
| Weighted Avg    | 0.93      | 0.94   | 0.93     | 72023   |

#### Feature Importance

| Feature                                | Importance |
|----------------------------------------|------------|
| Age                                    | 1339       |
| Dividends from Stocks                  | 860        |
| Capital Gains                          | 613        |
| Num Persons Worked for Employer        | 584        |
| Capital Losses                         | 562        |
| Weeks Worked in Year                   | 531        |
| Wage per Hour                          | 466        |
| Marital Stat Married                   | 181        |
| Class of Worker Self-Employed          | 169        |
| Class of Worker Private Sector         | 163        |
| Sex Male                               | 158        |
| Full or Part Time Employment Stat FTE  | 152        |
| Education College Graduate             | 151        |
| Migration Code Change in Reg Same Area | 133        |
| Tax Filer Stat Individual Filer        | 132        |

### Confusion Matrix

|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative| 3037               | 3149               |
| Actual Positive| 1172               | 64665              |


- True Negatives (TN): 3037
- False Positives (FP): 3149
- False Negatives (FN): 1172
- True Positives (TP): 64665

### Confusion Matrix Percentages

- True Negatives (TN): 4.22%
- False Positives (FP): 4.37%
- False Negatives (FN): 1.63%
- True Positives (TP): 89.78%

### Feature Correlation Analysis

| Feature                          | Correlation with Prediction | Mean Value When Predicted 1 | Mean Value When Predicted 0 |
|----------------------------------|-----------------------------|-----------------------------|-----------------------------|
| age                              | -0.030927                   | 44.587489                   | 46.923022                   |
| dividends_from_stocks            | -0.225940                   | 144.390863                  | 2227.591827                 |
| capital_gains                    | -0.312269                   | 166.634648                  | 7384.224994                 |
| num_persons_worked_for_employer  | -0.174581                   | 2.534226                    | 4.319553                    |
| capital_losses                   | -0.192327                   | 34.124384                   | 287.501544                  |

### Detailed Model Performance Metrics

- Accuracy: 0.9400
- True Positive Rate: 0.9822
- True Negative Rate: 0.4909
- Positive Predictive Value: 0.9536

### Model Configuration

- Number of training rounds: 100
- Model Parameters:
    - num_leaves: 98
    - learning_rate: 0.0993595777192241
    - feature_fraction: 0.5396455649123717
    - bagging_fraction: 0.9906308241132946
    - bagging_freq: 1
    - objective: binary
    - metric: binary_logloss
    - boosting_type: gbdt
    - verbose: 0

### Class Distribution in Training Data

- Class counts:
    - 1: 131088
    - 0: 12380
- Class proportions:
    - 1: 0.913709
    - 0: 0.086291

### Training History

- Train: `binary_logloss`: 0.13247794863000417
- Valid: `binary_logloss`: 0.1565673297515884

### Feature Importance Percentages (Top 10)

| Feature                                | Importance Percentage |
|----------------------------------------|------------------------|
| age                                    | 13.80                  |
| dividends_from_stocks                  | 8.87                   |
| capital_gains                          | 6.32                   |
| num_persons_worked_for_employer        | 6.02                   |
| capital_losses                         | 5.79                   |
| weeks_worked_in_year                   | 5.47                   |
| wage_per_hour                          | 4.80                   |
| marital_stat_married                   | 1.87                   |
| class_of_worker_self-employed          | 1.74                   |
| class_of_worker_private_sector         | 1.68                   |

### Prediction Threshold Analysis

| Threshold | Accuracy |
|-----------|----------|
| 0.1       | 0.928148 |
| 0.2       | 0.933202 |
| 0.3       | 0.937339 |
| 0.4       | 0.940019 |
| 0.5       | 0.940005 |
| 0.6       | 0.937631 |
| 0.7       | 0.929856 |
| 0.8       | 0.911931 |
| 0.9       | 0.862169 |

### Model Complexity Analysis

- Number of trees: 100
- Total number of leaves: 9800

### Prediction Confidence Analysis

| Prediction Confidence Interval | Actual Count | Mean |
|--------------------------------|--------------|------|
| (4.48e-05, 0.0676]             | 929          | 0.032293 |
| (0.0676, 0.134]                | 364          | 0.151099 |
| (0.134, 0.201]                 | 422          | 0.199052 |
| (0.201, 0.267]                 | 455          | 0.252747 |
| (0.267, 0.334]                 | 547          | 0.352834 |
| (0.334, 0.401]                 | 554          | 0.407942 |
| (0.401, 0.467]                 | 632          | 0.485759 |
| (0.467, 0.534]                 | 635          | 0.544882 |
| (0.534, 0.6]                   | 773          | 0.586028 |
| (0.6, 0.667]                   | 948          | 0.669831 |
| (0.667, 0.734]                 | 1210         | 0.720661 |
| (0.734, 0.8]                   | 1778         | 0.781215 |
| (0.8, 0.867]                   | 2771         | 0.840130 |
| (0.867, 0.933]                 | 5763         | 0.905258 |
| (0.933, 1.0]                   | 54242        | 0.987924 |
