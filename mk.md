## Variable Mapping

| Column Name                                 | Data Type | Required | Mapping |
|---------------------------------------------|-----------|----------|---------|
| age                                         | int64     | y        |         |
| class_of_worker                             | object    | y        | map     |
| detailed_industry_recode                    | object    | y        | map     |
| detailed_occupation_recode                  | object    | y        | map     |
| education                                   | object    | y        | map     |
| wage_per_hour                               | int64     | y        |         |
| enroll_in_edu_inst_last_wk                  | object    | y        |         |
| marital_stat                                | object    | y        | map     |
| major_industry_code                         | object    | y        |         |
| major_occupation_code                       | object    | y        |         |
| race                                        | object    | y        |         |
| hispanic_origin                             | object    | n        |         |
| sex                                         | object    | y        |         |
| member_of_a_labor_union                     | object    | y        |         |
| reason_for_unemployment                     | object    | y        |         |
| full_or_part_time_employment_stat           | object    | y        | map     |
| capital_gains                               | int64     | y        |         |
| capital_losses                              | int64     | y        |         |
| dividends_from_stocks                       | int64     | y        |         |
| tax_filer_stat                              | object    | y        | map     |
| region_of_previous_residence                | object    | y        |         |
| state_of_previous_residence                 | object    | n        |         |
| detailed_household_and_family_stat          | object    | y        | map     |
| detailed_household_summary_in_household     | object    | y        | map     |
| instance_weight                             | float64   | n        |         |
| migration_code_change_in_msa                | object    | y        | map     |
| migration_code_change_in_reg                | object    | y        |         |
| migration_code_move_within_reg              | object    | n        |         |
| live_in_this_house_1_year_ago               | object    | y        |         |
| migration_prev_res_in_sunbelt               | object    | y        |         |
| num_persons_worked_for_employer             | int64     | y        |         |
| family_members_under_18                     | object    | n        |         |
| country_of_birth_father                     | object    | y        |         |
| country_of_birth_mother                     | object    | y        |         |
| country_of_birth_self                       | object    | y        |         |
| citizenship                                 | object    | y        |         |
| own_business_or_self_employed               | object    | y        | map     |
| fill_inc_questionnaire_for_veterans_admin   | object    | y        |         |
| veterans_benefits                           | object    | y        |         |
| weeks_worked_in_year                        | int64     | y        |         |
| year                                        | int64     | y        |         |
| target                                      | object    | T        |         |



## Dropping Duplicates
There were 3229 duplicated records.. All belong to the -50k category


## Remapping
Remapping was done for perfrom feature engineering and grouping various 



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