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