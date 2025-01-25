from ML.rfc_v1 import classifier as classifier_rfc
from ML.lgbn_v3 import classifier as classifier_lgbm
from preprocessing import preprocess
from notebook_runner import run_notebook

RUN_PPEPROCESSING = False
RUN_EDA_NOTEBOOK = False
RUN_RFC_CLASSIFIER = False
RUN_LGMB_CLASSIFER = True
RUN_FALSEPOSITIVE_ANALYSIS = True

def main():

    '''
    this is the main function that performs
    1. pre-processing etl
    2. run ML classifier for light gbm
    3. run ML classifier for random forest
    '''

    #run pre-processing
    if RUN_PPEPROCESSING:
        preprocess()

    #run EDA notebooks for graphs and statistics
    if RUN_EDA_NOTEBOOK:
        notebook_path = 'Explore-Graph.ipynb'
        run_notebook(notebook_path)

    #run ML process for the Random Forest Classifier
    if RUN_RFC_CLASSIFIER:
        classifier_rfc()

    # # #run ML process for the Light GBM Classifier
    if RUN_LGMB_CLASSIFER:
        classifier_lgbm()
    
    if RUN_FALSEPOSITIVE_ANALYSIS:
        notebook_path = 'Explore-ML-FPostive.ipynb'
        run_notebook(notebook_path)

if __name__ == "__main__":
    main()