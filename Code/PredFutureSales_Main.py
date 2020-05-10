from PredFutureSales_Class import PredFutureSales
from config import *

if __name__ == "__main__":
    try:
        PFS = PredFutureSales(params, flowargs)
        PFS.readdata()
        PFS.createrawfeatures()
        PFS.featurepreprocessing()
        PFS.featureengineering()
        PFS.holdoutrunner(folds=params['folds'], shift=params['holdoutshift'])
        PFS.kagglesubmit()
        PFS.endrun()
    except Exception as e:
        PFS.flow.log_status(status="Failed", errormessage=e)