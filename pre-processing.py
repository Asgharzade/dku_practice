import os
import pandas as pd
from ydata_profiling import ProfileReport
import json

#load metadata
with open('data/processing/metadata.json', 'r') as f:
    definitions = json.load(f)

with open('data/processing/industry-recode.json', 'r') as f:
    industry_recode = json.load(f)

with open('data/processing/occupasion-recode.json', 'r') as f:
    occupasion_recode = json.load(f)

with open('data/processing/education-recode.json', 'r') as f:
    education_recoder = json.load(f)

with open('data/processing/martial-recode.json', 'r') as f:
    martial_recoder = json.load(f)

with open('data/processing/unemployment-recode.json', 'r') as f:
    unemployment_recoder = json.load(f)

with open('data/processing/typeemployment-recode.json', 'r') as f:
    typeemployment_recoder = json.load(f)

with open('data/processing/migrationwithin-recode.json', 'r') as f:
    migrationwithin_recoder = json.load(f)

with open('data/processing/broader_values_mapping.json', 'r') as f:
    broader_values_mapping = json.load(f)

with open('data/processing/class_worker_mapping.json', 'r') as f:
    class_worker_mapping = json.load(f)

with open('data/processing/household_status_mapping.json', 'r') as f:
    household_status_mapping = json.load(f)

with open('data/processing/msa_migration_mapping.json', 'r') as f:
    msa_migration_mapping = json.load(f)

with open('data/processing/tax_status_mapping.json', 'r') as f:
    tax_status_mapping = json.load(f)


def load_data(file_dir: str) -> pd.DataFrame:
    '''
    this function loads the data and applies the necessary cleaning steps
        input: file_dir: str: the path to the file
        output: df: pd.DataFrame: the cleaned dataframe
    '''

    #load the data
    df = pd.read_csv(file_dir, header=None, names=definitions.keys())
    
    # Remove duplicates and keep the first occurrence
    print(f"Number of duplicate rows removed: {df.shape[0] - df.drop_duplicates().shape[0]}")
    df = df.drop_duplicates()

    # cleaning
    # remove ignore col
    try: 
        df.drop(columns=['ignore'], inplace=True)
    except:
        pass

    # fix year"
    if df['year'].min() < 100:
        df['year'] = df['year'] + 1900

    # map industry code
    if df['detailed_industry_recode'].dtype == 'int64':
        df['detailed_industry_recode'] = df['detailed_industry_recode'].astype(str).map(industry_recode)

    # map occupation code
    if df['detailed_occupation_recode'].dtype == 'int64':
        df['detailed_occupation_recode'] = df['detailed_occupation_recode'].astype(str).map(occupasion_recode)

    if 'Advanced Degree' not in df['education'].unique():
        df['education'] = df['education'].map(education_recoder)

    if 'Married' not in df['marital_stat'].unique():
        df['marital_stat'] = df['marital_stat'].map(martial_recoder)

    if ' Other job loser' in df['reason_for_unemployment'].unique():
        df['reason_for_unemployment'] = df['reason_for_unemployment'].map(unemployment_recoder)

    if ' Not in labor force' in df['full_or_part_time_employment_stat'].unique():
        df['full_or_part_time_employment_stat'] = df['full_or_part_time_employment_stat'].map(typeemployment_recoder)

    df['migration_prev_res_in_sunbelt'] = df['migration_prev_res_in_sunbelt'].replace(' ?', 'Not in universe')

    df['migration_code_change_in_reg'] = df['migration_code_change_in_reg'].map(migrationwithin_recoder)

    # map values 

    df['migration_code_change_in_msa'] = df['migration_code_change_in_msa'].map(msa_migration_mapping)
    df['detailed_household_and_family_stat'] = df['detailed_household_and_family_stat'].map(household_status_mapping)
    df['migration_code_change_in_reg'].map(broader_values_mapping)
    df['tax_filer_stat'] = df['tax_filer_stat'].map(tax_status_mapping)
    df['class_of_worker'] = df['class_of_worker'].map(class_worker_mapping)

    # map additional values
    veteran_status_mapping = {
        2: 'Not a Veteran',
        1: 'Veteran',
        0: 'Not in universe'
    }
    df['veterans_benefits'] = df['veterans_benefits'].map(veteran_status_mapping)

    selfemployed_mapping = {
        2: 'No',
        1: 'Yes',
        0: 'Not in universe'
    }
    df['own_business_or_self_employed'] = df['own_business_or_self_employed'].map(selfemployed_mapping)

    # replacement values
    df['migration_prev_res_in_sunbelt']=df['migration_prev_res_in_sunbelt'].replace('Not in universe', 'Not in universe').replace(' Not in universe', 'Not in universe')
    df['live_in_this_house_1_year_ago']=df['live_in_this_house_1_year_ago'].replace(' Not in universe under 1 year old', 'Not in universe')

    # replacing the  target column
    df['target'] = df['target'].replace(' - 50000.', '1').replace(' 50000+.', '0')

    return df

data_input = ['data/raw/census_income_learn.csv', 'data/raw/census_income_test.csv']
for finename in data_input:
    print(f"Processing {finename}")
    df = load_data(finename)
    df.to_csv(f'data/processed/{finename.split("/")[-1]}', index=False)
    profile = ProfileReport(df, title=f"census data report for {finename.split("/")[-1]}")
    print(f"Processing {finename} completed")
    profile.to_file(f'report/processed/{finename.split("/")[-1]}.html')
    





