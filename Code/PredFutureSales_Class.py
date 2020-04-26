import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rfpimp import importances
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

from DeepFlow import DeepFlow


def createlag(data, col, lag, groupcols):
    return data.groupby(groupcols)[col].shift(lag).fillna(0).values

def addmonth(period, months):
    """
    Add, subtract months from a period in YYYYMM format
    """
    pdate = datetime.strptime(period, '%Y%m')
    ryear = pdate.year

    rmonth = pdate.month + months
    if rmonth > 12:
        rmonth -= 12
        ryear += 1

    return f"{ryear}{str(rmonth).zfill(2)}"


class PredFutureSales():
    def __init__(self, params, flowargs):
        self.params = params
        self.rundesc = flowargs['description']
        self.flow = DeepFlow(**flowargs)
        self.filename = f"exp_{self.flow.dfcurrentrun.ExpID[0]}.csv"
        self.imppath = f'../Artefacts/exp_{self.flow.dfcurrentrun.ExpID[0]}'

        print(f"Starting Experiment {self.flow.dfcurrentrun.ExpID[0]}")

    def readdata(self):
        """
        This function reads all the input data from the 
        input path mentioned in params 
        """
        ip = self.params['ip']
        self.df_train = pd.read_csv(f'{ip}/sales_train.csv')
        self.df_items = pd.read_csv(f'{ip}/items.csv')
        self.df_shops = pd.read_csv(f'{ip}/shops_en.csv')
        self.df_itemcat = pd.read_csv(f'{ip}/item_categories_en.csv')
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

        print("\nCreating calendar")
        
        years = pd.DataFrame({'Year':np.arange(2013, 2016, dtype=np.int32)})
        years['Key'] = 1

        months = pd.DataFrame({'Month':np.arange(1, 13, dtype=np.int32), 'Key':np.ones(12, dtype=np.int32)})

        cal = pd.merge(years, months, on='Key')
        del years, months

        cal['period'] = [f"{i}{str(j).zfill(2)}" for i, j in zip(cal.Year, cal.Month)]
        cal = cal[cal.period<'201511']

        print("Creating Raw Features")

        self.df_test['Key'] = 1
        self.df_test = pd.merge(self.df_test, self.df_items[['item_id', 'item_category_id']], on='item_id', how='left')
        print(f"Missing item categories : {self.df_test.item_category_id.isna().sum()}")

        del self.df_items

        calxkeys = pd.merge(self.df_test, cal, on='Key')

        calxkeys.drop(columns='Key', inplace=True)
        self.df_test.drop(columns='Key', inplace=True)

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
        self.df_test['period'] = '201511'
        self.df_test['Year'] = 2015
        self.df_test['Month'] = 11
        self.df_test['item_cnt_day'] = 0
        self.df_test['item_price'] = np.NaN

        rawfeatures = pd.concat([rawfeatures, self.df_test], axis=0, sort=False)
        rawfeatures.item_cnt_day.fillna(0, inplace=True)

        self.rawfeatures = rawfeatures.copy()
        del rawfeatures

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

    def _bagofwords(self, df, colname, idcol, min_df=3):
        """
        Applies bag of words to the specified column
        and concats with the id cols
        """
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df[colname])

        bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()).T
        bow['ngram'] = [len(i.split()) for i in bow.index]
        bow = bow.sort_values(by='ngram', ascending=False)
        bow = bow.drop(columns='ngram').drop_duplicates().T
        bow.columns = [f"{colname}_{i}" for i in bow.columns]

        df = pd.concat([df[idcol], bow], axis=1)

        return df

    def featureengineering(self):
        """
        This function does the following feature engineering
        1. Create lags of sales as specified in params
        2. Create interaction shopid_category_id feature
        3. Adds bag of words for shops
        4. Adds bag of words for categories
        5. Adds days since last sales
        """        
        print(f"Creating {self.params['laglist']} lags of sales")

        for lag in self.params['laglist']:
            self.rawfeatures[f"item_cnt_day_lag{lag}"] = createlag(self.rawfeatures, 'item_cnt_day', lag, self.params['mkey_cols'])

        print("Creating shop_categoryid interaction")
        self.rawfeatures['shop_category'] = [f"{i}_{j}" for i, j in zip(self.rawfeatures.shop_id, self.rawfeatures.item_category_id)]

        print("Adding bag of words for shops")
        shops_bow = self._bagofwords(self.df_shops, colname='shop_name_en', idcol='shop_id') 
        self.rawfeatures = pd.merge(self.rawfeatures, shops_bow, on='shop_id', how='left')

        print("Adding bag of words for categories")
        categories_bow = self._bagofwords(self.df_itemcat, colname='item_category_name_en', idcol='item_category_id') 
        self.rawfeatures = pd.merge(self.rawfeatures, categories_bow, on='item_category_id', how='left')

        print("Adding days since last sales")
        dfnonzerosales = self.rawfeatures[self.rawfeatures['item_cnt_day'] > 0]['shop_id', 'item_id', 'period']
        dfnonzerosales.rename(columns={'period':'lastsaleperiod'}, inplace=True)
        self.rawfeatures = pd.merge(self.rawfeatures, dfnonzerosales, on=self.params['mkey_cols'], how='left')
        self.rawfeatures['lastsaleperiod'].fillna(method='ffill', inplace=True)
        self.rawfeatures['lastsaleperiod'].fillna(0, inplace=True)
        self.rawfeatures['lastsaleperiod'] = createlag(self.rawfeatures, 'lastsaleperiod', 1, self.params['mkey_cols'])
        self.rawfeatures['months_since_sale'] = [0 if j==0 else 12*(int(i[:4]) - int(j[:4])) + (int(i[-2:]) - int(j[-2:])) 
            for i, j in zip(self.rawfeatures['period'], self.rawfeatures['lastsaleperiod'])]

        print(self.rawfeatures[['shop_id', 'item_id', 'period', 'lastsaleperiod', 'months_since_sale']].sample(20))
        
        self.rawfeatures.drop(columns='lastsaleperiod', inplace=True)

        print(f"raw features shape after feature engineering : {self.rawfeatures.shape}")
        print(f"any missing cols? : {self.rawfeatures.columns[self.rawfeatures.isnull().any()].tolist()}")
    
    def _timeseriessplit(self, trainstart='201301', holdoutstart='201511', holdoutmonths = 1, final=False):
        """
        Does a time series split of the data
        1. Train set will consist of data in range [trainstart, holdoutstart)
        2. Holdout set will consist of data in range [holdoutstart,  holdoutstart + holdoutmonths)
        3. Test set is in month 201511
        """
        print(f"Train Start : {trainstart}") 
        self.df_train = self.rawfeatures[(self.rawfeatures.period >= trainstart) & (self.rawfeatures.period < holdoutstart)]

        if not final:
            holdoutend = addmonth(holdoutstart, holdoutmonths)
            self.df_holdout = self.rawfeatures[(self.rawfeatures.period >= holdoutstart) & (self.rawfeatures.period < holdoutend)]

            print(f"Holdout Start : {holdoutstart}")
            print(f"Holdout End : {holdoutend}")


    def _train(self):
        """
        Trains model on the train set and evaluates accuracy of holdout set
        """
        X_train = self.df_train.drop(columns=self.params['ignorecols'])
        y_train = self.df_train[self.params['targetcol']]

        self.params['Pipeline'].fit(X_train, y_train)

    def _predict(self, X):
        X = X.drop(columns=self.params['ignorecols'])
        return np.expm1(self.params['Pipeline'].predict(X))

    def _score(self, pred, actuals):
        return np.sqrt(mean_squared_error(pred, actuals))        

    def _permutationimportance(self):
        """
        Finds the permutation importance and saves the importance.csv file
        in the Artefacts/exp_num folder
        """
        X_valid = self.df_holdout.drop(columns=self.params['ignorecols'])
        y_valid = self.df_holdout[self.params['targetcol']]

        imp = importances(
            self.params['Pipeline'], 
            X_valid,
            y_valid
        ).reset_index()

        ### sum of importances should sum to 1
        imp['Importance'] = imp['Importance']/sum(imp['Importance'])

        return imp

    def holdoutrunner(self, folds = 5, shift = 1):
        """
        This function runs n-holdouts in time series fashion
        shifting the holdout period by shift months
        """
        trainstart = self.params['trainstart']
        holdoutstart = self.params['holdstart']
        scores = {}
        dfimp = pd.DataFrame(columns=['Holdout', 'Feature', 'Importance'])

        for fold in range(1, folds+1):
            print(f"\nFold {fold}:{folds}")

            print("\nTime Series Split")
            self._timeseriessplit(trainstart=trainstart, holdoutstart=holdoutstart, holdoutmonths=self.params['holdoutmonths'])
            
            print(f"\nTraining")
            self._train()
            
            print(f"\nPredicting")
            pred = self._predict(self.df_holdout)

            y_valid = np.expm1(self.df_holdout[self.params['targetcol']])
            score = self._score(pred, y_valid)
            print(f"\nRMSE : {score}")
            scores[(fold, holdoutstart)] = score

            print("\nCalculating feature importance")
            imp = self._permutationimportance()
            imp['Holdout'] = fold
            dfimp = pd.concat([dfimp, imp], axis=0)

            holdoutstart = addmonth(holdoutstart, shift)

        self.flow.log_param("Holdout Scores", scores)
        self.flow.log_score("Average RMSE", np.mean(list(scores.values())))
        self.flow.log_imp(dfimp, self.imppath)

    def _finalize(self):
        self._timeseriessplit(trainstart=self.params['trainstart'], final=True)
        self._train()

    def kagglesubmit(self):
        """
        Save submission file in submissions folder and print command line argument
        to be executed if we want to submit the results to kaggle 
        """
        kagglescore = np.NaN

        print("Submit to kaggle? : Y/N")

        if input().lower() == 'y':
            self._finalize()

            df_test = self.rawfeatures[self.rawfeatures.period=='201511']

            df_test['item_cnt_month'] = self._predict(df_test)

            submission = df_test.loc[: , ['ID', 'item_cnt_month']]

            submission['item_cnt_month'] = submission['item_cnt_month'].apply(lambda x: 0 if x<0 else (20 if x>20 else x))

            submission.to_csv(self.params['op']+self.filename, index=False)
            print(f"submission file : {self.params['op']+self.filename}")

            print(f"Run following command on terminal in submissions folder : ")
            print(f"kaggle competitions submit -c competitive-data-science-predict-future-sales -f {self.filename} -m '{self.rundesc}'")

            # print("Kaggle score ,press enter if you dont want to submit :")
            ### add multiple scores support to DeepFlow
        else:
            print("We'll submit the next run")
    
    def endrun(self):
        self.flow.end_run()
