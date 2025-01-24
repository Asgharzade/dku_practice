from ML.rfc_v1 import classifier as classifier_rfc
from ML.lgbn_v3 import classifier as classifier_lgbn
from preprocessing import preprocess
from notebook import run_notebook

def main():

    '''
    this is the main function that performs
    1. pre-processing etl
    2. run ML classifier for light gbm
    3. run ML classifier for random forest
    '''

    #run pre-processing
    preprocess()

    #run EDA notebooks for graphs and statistics
    notebook_path = 'Explore-Graph.ipynb'
    run_notebook(notebook_path)

    # #run ML process for the Random Forest Classifier
    classifier_rfc()

    # # #run ML process for the Light GBM Classifier
    classifier_lgbn()

if __name__ == "__main__":
    main()