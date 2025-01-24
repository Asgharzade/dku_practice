from ML.rfc_v1 import classifier as classifier_rfc
from ML.lgbn_v3 import classifier as classifier_lgbn
from preprocessing import preprocess

def main():

    '''
    this is the main function that performs
    1. pre-processing etl
    2. run ML classifier for light gbm
    3. run ML classifier for random forest
    '''

    #run pre-processing
    preprocess()

    # #run ML process for the Random Forest Classifier
    # classifier_rfc()

    # #run ML process for the Light GBM Classifier
    classifier_lgbn()

if __name__ == "__main__":
    main()
