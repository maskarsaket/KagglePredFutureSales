from PredFutureSales_Class import PredFutureSales
from config import *

if __name__ == "__main__":
    PFS = PredFutureSales(params, flowargs)
    PFS.readdata()
    PFS.createrawfeatures()
    PFS.featurepreprocessing()
    PFS.featureengineering()
    PFS.holdoutrunner(folds=params['folds'], shift=params['holdoutshift'])
    PFS.finalize()
    PFS.kagglesubmit()
    PFS.endrun()