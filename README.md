# A ML approach for low/high income classification Model

# Requirements

## System Requirements
- Python 3.8 or higher installed on your system.

## Python Requirements
Preferably, install packages in a virtual environment.


## Run the Project
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

# Data Preprocessing for Census Data

This document outlines the steps taken to preprocess the census data for analysis. 
The preprocessing steps include variable mapping, dropping duplicates, remapping, post-processing, under-sampling,

## Steps

1. Variable Mapping
2. Dropping Duplicates
3. Remapping
4. Post Processing
5. Under Sampling
6. Over Sampling











## Variable Mapping

| Column Name                                 | Data Type | Required | Mapping | Post Processing|
|---------------------------------------------|-----------|----------|---------|----------------|
| age                                         | int64     | y        |         |ok, only over 18    |
| class_of_worker                             | object    | y        | map     |ok, keep only 5 cats|
| detailed_industry_recode                    | object    | y        | map     |no,high corr with class_of_worker|
| detailed_occupation_recode                  | object    | y        | map     |no,high corr with class_of_worker|
| education                                   | object    | y        | map     |ok|
| wage_per_hour                               | int64     | y        |         |ok|
| enroll_in_edu_inst_last_wk                  | object    | y        |         |ok|
| marital_stat                                | object    | y        | map     |ok|
| major_industry_code                         | object    | y        |         |ok|
| major_occupation_code                       | object    | y        |         |ok|
| race                                        | object    | y        |         |ok|
| hispanic_origin                             | object    | n        |         |not relative|
| sex                                         | object    | y        |         |ok, might be sensitive|
| member_of_a_labor_union                     | object    | y        |         |ok, just keep 'Yes'|
| reason_for_unemployment                     | object    | y        |         |ok|
| full_or_part_time_employment_stat           | object    | y        | map     |ok|
| capital_gains                               | int64     | y        |         |ok|
| capital_losses                              | int64     | y        |         |ok|
| dividends_from_stocks                       | int64     | y        |         |ok|
| tax_filer_stat                              | object    | y        | map     |ok|
| region_of_previous_residence                | object    | y        |         |ok. just four cats|
| state_of_previous_residence                 | object    | n        |         |no, too many cats|
| detailed_household_and_family_stat          | object    | y        | map     |ok|
| detailed_household_summary_in_household     | object    | y        | map     |no, too many cats|
| instance_weight                             | float64   | n        |         |no, instructed|
| migration_code_change_in_msa                | object    | y        | map     |no, high corr with move_reg|
| migration_code_change_in_reg                | object    | y        |         |ok, only two cats|
| migration_code_move_within_reg              | object    | n        |         |no, high corr with move_reg|
| live_in_this_house_1_year_ago               | object    | y        |         |ok|
| migration_prev_res_in_sunbelt               | object    | y        |         |ok|
| num_persons_worked_for_employer             | int64     | y        |         |ok|
| family_members_under_18                     | object    | n        |         |ok|
| country_of_birth_father                     | object    | y        |         |no, high corr with citizenship|
| country_of_birth_mother                     | object    | y        |         |no, high corr with citizenship|
| country_of_birth_self                       | object    | y        |         |no, high corr with citizenship|
| citizenship                                 | object    | y        |         |ok|
| own_business_or_self_employed               | object    | y        | map     |ok|
| fill_inc_questionnaire_for_veterans_admin   | object    | y        |         |no, using veterans_benefits instead|
| veterans_benefits                           | object    | y        |         |ok|
| weeks_worked_in_year                        | int64     | y        |         |ok|
| year                                        | int64     | y        |         |not related|
| target                                      | object    | T        |         |Target|



## Dropping Duplicates
There were 3229 duplicated records.. All belong to the -50k category


## Remapping
Remapping was done for perfrom feature engineering and grouping various.
The number of groups 



## Post Processing




## Under Sampling
For census data, which typically involves categorical features and inherent population patterns, I'd recommend using either:

One-Sided Selection (OSS) would be the best choice because:


Census data often contains natural redundancy due to similar demographic patterns
OSS preserves important boundary cases while removing redundant majority samples
It works well with categorical variables common in census data
The two-step process (Tomek Links + CNN) helps maintain important demographic patterns
It's less likely to distort the natural distribution of population segments


Neighborhood Cleaning Rule (NCR) would be the second best choice because:


It's effective at identifying and removing noise while preserving structure
Works well with mixed numerical and categorical features
Maintains local demographic patterns in the data
Less aggressive than random undersampling

Random undersampling would be too destructive for census data, as it might eliminate entire demographic groups or important patterns. Similarly, Tomek Links alone might be too focused on boundary cases, missing broader population patterns.
The choice between OSS and NCR might depend on:

The specific imbalance ratio in your census data
The importance of boundary cases in your classification problem
The computational resources available (OSS can be more computationally intensive)
The specific demographic patterns you need to preserve

Would you like me to explain how to implement either of these methods for your census data?


## Over Sampling
For census data, SMOTE-NC (SMOTE for Nominal and Continuous features) is generally considered the best oversampling choice because:

Feature Type Handling


Census data typically contains both numerical features (age, income) and categorical features (occupation, education level)
SMOTE-NC is specifically designed to handle this mixed data type scenario
It uses different synthesis methods for numerical vs categorical variables
For categorical features, it uses a probability-based approach rather than simple interpolation


Preservation of Data Characteristics


Maintains the relationship between categorical variables
Preserves the natural distribution of categorical values
Respects the discrete nature of categorical variables instead of creating unrealistic interpolations


Advantages over alternatives:


Regular SMOTE would fail on categorical features
Random oversampling would create exact duplicates, leading to overfitting
ADASYN might create too much noise in categorical variables
Borderline-SMOTE might not capture the full demographic patterns

However, there are some considerations:

You need to properly encode categorical variables
The synthetic samples should be validated to ensure they represent realistic demographic combinations
Computational cost increases with many categorical variables
May need domain expertise to validate the synthetic samples make sense in a census context