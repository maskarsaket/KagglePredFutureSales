import gc
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rfpimp import importances
from sklearn.metrics import mean_squared_error

from DeepFlow.deepflow import DeepFlow


def createlag(data, col, lag, groupcols):
    return data.groupby(groupcols)[col].shift(lag).fillna(0).values

class PredFutureSales():
    def __init__(self, params, flowargs):
        self.rundesc = flowargs['description']
        self.flow = DeepFlow(**flowargs)
        self.filename = f"exp_{self.flow.dfcurrentrun.ExpID[0]}.csv"
        self.imppath = f'../Artefacts/exp_{self.flow.dfcurrentrun.ExpID[0]}'

    def readdata(self):
        """
        This function reads all the input data from the 
        input path mentioned in params 
        """
        ip = self.params['ip']
        self.df_train = pd.read_csv(f'{ip}/sales_train.csv')
        self.df_items = pd.read_csv(f'{ip}/items.csv')
        # df_shops = pd.read_csv(f'{ip}/shops.csv')
        # df_itemcat = pd.read_csv(f'{ip}/item_categories.csv')
        self.df_test = pd.read_csv(f'{ip}/test.csv')

    def createrawfeatures(self):
        """
        This function aggregates the sales to monthly level, adds data points 
        with zero sales for months with no sale
        """
        df_trainm = self.df_train.copy()
        
        df_trainm['date'] = pd.to_datetime(df_trainm['date'], format='%d.%m.%Y')
        df_trainm['Year'] = df_trainm['date'].dt.year
        df_trainm['Month'] = df_trainm['date'].dt.month

        df_trainm['period'] = [str(i) + str(j).zfill(2) for i, j in zip(df_trainm.Year, df_trainm.Month)]

        groupcols = ['shop_id','item_id','period','Year','Month']
        df_trainm = df_trainm.groupby(groupcols, as_index=False).agg({
            'item_cnt_day':'sum',
            'item_price':'mean'
        })

        del self.df_train
        gc.collect()

        print("\nCreating calendar")
        
        years = pd.DataFrame({'Year':np.arange(2013, 2016, dtype=np.int32)})
        years['Key'] = 1

        months = pd.DataFrame({'Month':np.arange(1, 13, dtype=np.int32), 'Key':np.ones(12, dtype=np.int32)})

        cal = pd.merge(years, months, on='Key')
        del years, months

        cal['period'] = [f"{i}{str(j).zfill(2)}" for i, j in zip(cal.Year, cal.Month)]
        cal = cal[cal.period<'201511']

        print("Creating Raw Features")

        df_test['Key'] = 1
        df_test = pd.merge(df_test, df_items[['item_id', 'item_category_id']], on='item_id', how='left')
        print(f"Missing item categories : {df_test.item_category_id.isna().sum()}")

        calxkeys = pd.merge(df_test, cal, on='Key')

        calxkeys.drop(columns='Key', inplace=True)
        df_test.drop(columns='Key', inplace=True)

        rawfeatures = pd.merge(calxkeys, df_trainm,
                            on=groupcols, how='left')

        del calxkeys

        print("Removing rows for sales before first point of sales per mkey")
        dfmin = df_trainm.groupby(self.params['mkey_cols'], as_index=False).agg({'period':'min'}).rename(columns={'period':'minperiod'})

        rawfeatures = pd.merge(rawfeatures, dfmin, on=self.params['mkey_cols'], how='left')

        del dfmin

        ### remove rows of sales before first sale date
        rawfeatures = rawfeatures[rawfeatures.period >= rawfeatures.minperiod]
        rawfeatures.drop(columns='minperiod', inplace=True)

        print("Defining vaiables for test set and concatting with rawfeatures to create lags")
        df_test['period'] = '201511'
        df_test['Year'] = 2015
        df_test['Month'] = 11
        df_test['item_cnt_day'] = 0
        df_test['item_price'] = np.NaN

        rawfeatures = pd.concat([rawfeatures, df_test], axis=0, sort=False)
        rawfeatures.item_cnt_day.fillna(0, inplace=True)

        self.rawfeatures = rawfeatures

    def addprice(self):
        """
        This function will treat missing values of price
        First do ffill, then median by category, then overall median
        """
        raise NotImplementedError

    def featurepreprocessing(self):
        """
        1. Clips sales in the range [0, 20]
        2. Performs log transform of sales
        3. Converts categorical columns to type category
        """
        for col in self.params['numericcols']:
            self.rawfeatures[col] = self.rawfeatures[col].apply(lambda x : 0 if x<0 else (20 if x>20 else x))
            self.rawfeatures[col] = np.log1p(self.rawfeatures[col])

        for col in self.params['categoricalcols']:
            self.rawfeatures[col] = self.rawfeatures[col].astype('category')

    def featureengineering(self):
        """
        This function does the following feature engineering
        1. Create lags of sales as specified in params
        """        
        ### Creating lags of sales
        print(f"Creating {self.params['laglist']} lags of sales")

        for lag in self.params['laglist']:
            self.rawfeatures[f"item_cnt_day_lag{lag}"] = createlag(self.rawfeatures, 'item_cnt_day', lag, self.params['mkey_cols'])
            print(f"Created lag {lag}")
    
    def timeseriessplit(self, trainstart='201301', holdoutstart='201510', holdoutmonths = 5):
        """
        Does a time series split of the data
        1. Train set will consist of data in range [trainstart, holdoutstart)
        2. Holdout set will consist of data in range [holdoutstart,  holdoutstart + holdoutmonths)
        3. Test set is in month 201511
        """
        self.df_train = self.rawfeatures[(rawfeatures.period >= trainstart) & (self.rawfeatures.period < holdoutstart)]

        ### find holdout end month
        holdstartdate = datetime.strptime(holdoutstart, '%Y%m')
        holdendyear = holdstartdate.year

        holdendmonth = holdstartdate.month + holdoutmonths
        if holdendmonth > 12:
            holdendmonth-= 12
            holdendyear += 1

        holdoutend = f"{holdendyear}{str(holdendmonth).zfill(2)}" 

        self.df_holdout = self.rawfeatures[(self.rawfeatures.period >= holdoutstart) & (self.rawfeatures.period) < holdoutend]
        
        self.df_test = self.rawfeatures[self.rawfeatures.period=='201511']

        del self.rawfeatures

    def train(self):
        """
        Trains model on the train set and evaluates accuracy of holdout set
        """
        X_train = self.df_train.drop(columns=self.params['ignorecols'])
        y_train = self.df_train[self.params['targetcol']]

        X_valid = self.df_holdout.drop(columns=self.params['ignorecols'])
        y_valid = np.expm1(df_holdout[self.params['targetcol']])

        print("Training Model")
        self.params['Pipeline'].fit(X_train, y_train)

        print("Predicting")        
        pred = np.expm1(self.params['Pipeline'].predict(X_valid))

        holdouterror = np.sqrt(mean_squared_error(pred, y_valid))
        
        print(f"Holdout error : {holdouterror}")

        self.flow.log_score('RMSE', holdouterror, 4)

    def permutationimportance(self):
        """
        Finds the permutation importance and saves the importance.csv file
        in the Artefacts/exp_num folder
        """
        print("Permuting for feature importance")
        ### Save permutation importance
        imp = importances(
            self.params['Pipeline'], 
            df_holdout.drop(columns=self.params['Pipeline']),
            df_holdout[self.params['targetcol']]
        ).reset_index()

        ### sum of importances should sum to 1
        imp['Importance'] = imp['Importance']/sum(imp['Importance'])

        self.flow.log_imp(imp, self.imppath)

    def kagglesubmit(self):
        """
        Save submission file in submissions folder and print command line argument
        to be executed if we want to submit the results to kaggle 
        """
        kagglescore = np.NaN

        print("Submit to kaggle? : Y/N")

        if input().lower() == 'y':
            X_test = self.df_test.drop(columns=self.params['ignorecols'])
            self.df_test['item_cnt_month'] = np.expm1(self.params['Pipeline'].predict(X_test))

            submission = self.df_test.loc[: , ['ID', 'item_cnt_month']]

            submission['item_cnt_month'] = submission['item_cnt_month'].apply(lambda x: 0 if x<0 else (20 if x>20 else x))

            submission.to_csv(self.params['op']+self.filename, index=False)
            print(f"submission file : {self.params['op']+self.filename}")

            print(f"Run following command on terminal in submissions folder : ")
            print(f"kaggle competitions submit -c competitive-data-science-predict-future-sales -f {filename} -m '{self.rundesc}'")

            print("Kaggle score ,press enter if you dont want to submit :")
            ### add multiple scores support to DeepFlow
        else:
            print("We'll submit the next run")
    
    def endrun(self):
        self.flow.end_run()
