import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


features_ok = [
    'age',
    'class_of_worker',
    'education',
    'wage_per_hour',
    'enroll_in_edu_inst_last_wk',
    'marital_stat',
    'major_industry_code',
    'race',
    'sex',
    'member_of_a_labor_union',
    'reason_for_unemployment',
    'full_or_part_time_employment_stat',
    'capital_gains',
    'capital_losses',
    'dividends_from_stocks',
    'tax_filer_stat',
    'region_of_previous_residence',
    'detailed_household_and_family_stat',
    'migration_code_change_in_reg',
    'live_in_this_house_1_year_ago',
    'migration_prev_res_in_sunbelt',
    'num_persons_worked_for_employer',
    'family_members_under_18',
    'citizenship',
    'own_business_or_self_employed',
    'veterans_benefits',
    'weeks_worked_in_year',
    'target'
]


bin_cols_to_keep = [
    'class_of_worker_government',
    'class_of_worker_not_employed',
    'class_of_worker_private_sector',
    'class_of_worker_self-employed',
    'education_advanced_degree',
    'education_below_high_school',
    'education_college_graduate',
    'education_high_school_graduate',
    'education_some_college',
    'enroll_in_edu_inst_last_wk__college_or_university',
    'enroll_in_edu_inst_last_wk__high_school',
    'marital_stat_divorced',
    'marital_stat_married',
    'marital_stat_never_married',
    'marital_stat_separated',
    'marital_stat_widowed',
    'major_industry_code__agriculture',
    'major_industry_code__armed_forces',
    'major_industry_code__business_and_repair_services',
    'major_industry_code__communications',
    'major_industry_code__construction',
    'major_industry_code__education',
    'major_industry_code__entertainment',
    'major_industry_code__finance_insurance_and_real_estate',
    'major_industry_code__forestry_and_fisheries',
    'major_industry_code__hospital_services',
    'major_industry_code__manufacturing-durable_goods',
    'major_industry_code__manufacturing-nondurable_goods',
    'major_industry_code__medical_except_hospital',
    'major_industry_code__mining',
    'major_industry_code__other_professional_services',
    'major_industry_code__personal_services_except_private_hh',
    'major_industry_code__private_household_services',
    'major_industry_code__public_administration',
    'major_industry_code__retail_trade',
    'major_industry_code__social_services',
    'major_industry_code__transportation',
    'major_industry_code__utilities_and_sanitary_services',
    'major_industry_code__wholesale_trade',
    'race__amer_indian_aleut_or_eskimo',
    'race__asian_or_pacific_islander',
    'race__black',
    'race__white',
    'sex__male',
    'member_of_a_labor_union__no',
    'member_of_a_labor_union__yes',
    'reason_for_unemployment_job_leaver',
    'reason_for_unemployment_job_loser',
    'reason_for_unemployment_new_entrant',
    'reason_for_unemployment_re-entrant',
    'full_or_part_time_employment_stat_fte',
    'full_or_part_time_employment_stat_not_employed',
    'full_or_part_time_employment_stat_pte',
    'tax_filer_stat_individual_filer',
    'tax_filer_stat_non-filer',
    'region_of_previous_residence__abroad',
    'region_of_previous_residence__midwest',
    'region_of_previous_residence__northeast',
    'region_of_previous_residence__south',
    'region_of_previous_residence__west',
    'detailed_household_and_family_stat_child',
    'detailed_household_and_family_stat_extended_family',
    'detailed_household_and_family_stat_primary_householder',
    'migration_code_change_in_reg_different_area',
    'migration_code_change_in_reg_same_area',
    'live_in_this_house_1_year_ago__no',
    'live_in_this_house_1_year_ago__yes',
    'migration_prev_res_in_sunbelt__no',
    'migration_prev_res_in_sunbelt__yes',
    'family_members_under_18__both_parents_present',
    'family_members_under_18__father_only_present',
    'family_members_under_18__mother_only_present',
    'family_members_under_18__neither_parent_present',
    'citizenship_foreign',
    'citizenship_native',
    'own_business_or_self_employed_no',
    'own_business_or_self_employed_yes',
    'veterans_benefits_not_a_veteran',
    'veterans_benefits_veteran'
]



def post_processing(datadir: str,
                    features_ok = features_ok,
                    bin_cols_to_keep = bin_cols_to_keep) -> tuple:
    """
    Load the data, select the features, and split the data into training and testing sets.
        input: datadir (str) : path to the data
        output: tuple : X_train, X_test, y_train, y_test
    """
    
    # Load the data
    df = pd.read_csv(datadir)

    # Select the features
    df_selected = df[features_ok]

    # get numerical columns (non-object)
    numeric_cols = df_selected.select_dtypes(exclude=['object']).columns
    print(f"Number of numeric columns: {len(numeric_cols)}")
    df_numeric = df_selected[numeric_cols]

    # get categorical columns
    categorical_cols = df_selected.select_dtypes(include=['object']).columns
    df_binary = pd.get_dummies(df_selected[categorical_cols], drop_first=False)
    df_binary.columns = [x.lower() for x in df_binary.columns.str.replace(' ', '_')]
    print(f"Original shape: {df_selected.shape}")
    print(f"Binary shape: {df_binary.shape}")
    # select only the bin columns that are in the list of columns to keep
    df_binary = df_binary[bin_cols_to_keep]

    # merge the two dataframes
    df_prepared = pd.concat([df_numeric, df_binary], axis=1)

    # Split the data
    y = df_prepared['target']
    X = df_prepared.drop('target', axis=1)
    return X, y