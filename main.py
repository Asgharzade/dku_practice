from ML.rfc_v1 import classifier as classifier_rfc
from ML.lgbn_v3 import classifier as classifier_lgbm
from preprocessing import preprocess
from notebook_runner import run_notebook
from gl_downloader import download_from_gdrive

RUN_GOOGLEDR_DOWNLOADER = True
RUN_PPEPROCESSING = True
RUN_EDA_NOTEBOOK = True
RUN_RFC_CLASSIFIER = True
RUN_LGMB_CLASSIFER = True
RUN_FALSEPOSITIVE_ANALYSIS = True

file_links = [
    'https://drive.google.com/file/d/1JNprlyDTnj_FWS4s7Kqj_DK9c9e8paee/view?usp=drive_link',
    'https://drive.google.com/file/d/1Odd-l1rkvlyGnR8TCpqEDMaarPbv9v-D/view?usp=drive_link'
    ]

def main():

    '''
    this is the main function that performs
    0. download from google drive 
    1. pre-processing etl
    2. run ML classifier for light gbm
    3. run ML classifier for random forest
    4. perform false positive analysis
    '''

    #run google drive downloader
    if RUN_GOOGLEDR_DOWNLOADER:
        download_from_gdrive(file_links)

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