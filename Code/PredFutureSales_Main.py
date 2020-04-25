from PredFutureSales_Class import PredFutureSales
from config import *

if __name__ == "__main__":
    PFS = PredFutureSales(params, flowargs)
    PFS.readdata()
    PFS.createrawfeatures()
    PFS.featurepreprocessing()
    PFS.featureengineering()
    PFS.timeseriessplit(trainstart='201401')
    PFS.train()
    PFS.permutationimportance()
    PFS.kagglesubmit()
    PFS.endrun()