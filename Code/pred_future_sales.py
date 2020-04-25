import pandas as pd
import numpy as np
import os
import gc

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
# from lightgbm import LGBMRegressor

from joblib import Parallel, delayed

from DeepFlow.deepflow import DeepFlow

### define params
ip = '../data'
op = '../submissions/'
seed = 123
laglist = list(range(1, 13))
ignorecols = ['ID', 'item_cnt_day', 'period','item_price']
targetcol = 'item_cnt_day'

flowargs = {
    'projectname' : 'Kaggle - predict future sales',
    'runmasterfile' : '../runmaster.csv',
    'description' : 'Trying Gradient Boosting Regressor',
    'benchmark' : 1,
    'parentID' : 8
}

print(flowargs)

flow = DeepFlow(**flowargs)

filename = f"exp_{flow.dfcurrentrun.ExpID[0]}.csv"

# ### Reading all input data
df_train = pd.read_csv(f'{ip}/sales_train.csv')
# df_items = pd.read_csv(f'{ip}/items.csv')
# df_shops = pd.read_csv(f'{ip}/shops.csv')
# df_itemcat = pd.read_csv(f'{ip}/item_categories.csv')

df_test = pd.read_csv(f'{ip}/test.csv')

print(f'\ntrain shape : {df_train.shape}')
print(f'test shape : {df_test.shape}')
# print(f'items shape : {df_items.shape}')
# print(f'shops shape : {df_shops.shape}')
# print(f'categories shape : {df_itemcat.shape}')


# ### Agg to monthly level
df_trainm = df_train.copy()

df_trainm['date'] = pd.to_datetime(df_trainm['date'], format='%d.%m.%Y')
df_trainm['Year'] = df_trainm['date'].dt.year
df_trainm['Month'] = df_trainm['date'].dt.month

df_trainm['period'] = [str(i) + str(j).zfill(2) for i, j in zip(df_trainm.Year, df_trainm.Month)]

df_trainm = df_trainm.groupby(['shop_id','item_id','period','Year','Month'], as_index=False).agg({
    'item_cnt_day':'sum',
    'item_price':'mean'
})

print(f"Monthly level train shape : {df_trainm.shape}")

trainkeys = set(df_trainm['shop_id'].astype(str) + '_' + df_trainm['item_id'].astype(str))
testkeys = set(df_test['shop_id'].astype(str) + '_' + df_test['item_id'].astype(str))
newkeys = testkeys.difference(trainkeys)
print(f"{len(newkeys)} mkeys not present in train set")

del df_train, trainkeys, testkeys, newkeys
gc.collect()

print("\nCreating calendar")
years = pd.DataFrame({'Year':np.arange(2013, 2016, dtype=np.int32)})
years['Key'] = 1
months = pd.DataFrame({'Month':np.arange(1, 13, dtype=np.int32), 'Key':np.ones(12, dtype=np.int32)})

cal = pd.merge(years, months, on='Key')
cal['period'] = [f"{i}{str(j).zfill(2)}" for i, j in zip(cal.Year, cal.Month)]
cal = cal[cal.period<'201511']
print("Creating Raw Features")

del years, months

df_test['Key'] = 1

calxkeys = pd.merge(df_test, cal, on='Key')

calxkeys.drop(columns='Key', inplace=True)
df_test.drop(columns='Key', inplace=True)

rawfeatures = pd.merge(calxkeys, df_trainm,
                       on=['shop_id', 'item_id', 'Year', 'Month', 'period'], how='left')

del calxkeys

print("Removing rows for sales before first point of sales per mkey")
dfmin = df_trainm.groupby(['shop_id', 'item_id'], as_index=False).agg({'period':'min'}).rename(columns={'period':'minperiod'})

rawfeatures = pd.merge(rawfeatures, dfmin, on=['shop_id', 'item_id'], how='left')

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

if 'item_price' not in ignorecols:
    def ffillparallel(data):
        df = data.copy()
        df['item_price'].fillna(method='ffill', inplace=True)
        return df

    res = Parallel(n_jobs=-1, verbose=5)(delayed(ffillparallel)(group) for _, group in rawfeatures.groupby(['shop_id', 'item_id']))
    rawfeatures = pd.concat(res)
    del res

### define categorical features
cat_feat = ['shop_id', 'item_id', 'Year', 'Month']
numeric_cols = ['item_price', 'item_cnt_day']

### Log transform and clip sales before creating lags
for col in numeric_cols:
    rawfeatures[col] = rawfeatures[col].apply(lambda x : 0 if x<0 else (20 if x>20 else x))
    rawfeatures[col] = np.log1p(rawfeatures[col])

for col in cat_feat:
    rawfeatures[col] = rawfeatures[col].astype('category')

### Creating lags of sales
print(f"Creating {laglist} lags of sales")
print(f'Total jobs = {df_test.shape[0]}')

def createlag(data, col, lag, groupcols):
    return data.groupby(groupcols)[col].shift(lag).fillna(0).values

def applyParallel(data, func, n_jobs, **kwargs):
    res = Parallel(n_jobs=n_jobs, verbose=5, batch_size=10000)(delayed(func)(group, **kwargs) for _, group in data.groupby(kwargs['groupcols']))
    return pd.concat(res).values

for lag in laglist:
    rawfeatures[f"item_cnt_day_lag{lag}"] = createlag(rawfeatures, 'item_cnt_day', lag, ['shop_id', 'item_id'])
    print(f"Created lag {lag}")

print(rawfeatures.head(2))

### time series split instead of train test split
df_train = rawfeatures[(rawfeatures.period < '201510') & (rawfeatures.period >= '201401')]
df_holdout = rawfeatures[rawfeatures.period=='201510']
df_test = rawfeatures[rawfeatures.period=='201511']

print("train set : ")
print(df_train.head(2))

print("holdout set : ")
print(df_holdout.head(2))

del rawfeatures

### Make pipeline
pipe = make_pipeline(GradientBoostingRegressor())

print("Fitting Model")
pipe.fit(df_train.drop(columns=ignorecols), df_train[targetcol])

print("Predicting")
pred = pipe.predict(df_holdout.drop(columns=ignorecols))

holdouterror = np.sqrt(mean_squared_error(np.expm1(pred), np.expm1(df_holdout[targetcol])))
print(f"Holdout error : {holdouterror}")

flow.log_score('RMSE', holdouterror, 4)

# print("Permuting for feature importance")
# ### Save permutation importance
# imp = pd.DataFrame({
#     'features': df_train.drop(columns=ignorecols).columns,
#     'importance':permutation_importance(pipe, X=df_holdout.drop(columns=ignorecols), y=df_holdout[targetcol].values, scoring='neg_root_mean_squared_error', n_jobs=-1, n_repeats=1).importances_mean
# })
# imp = imp.sort_values(by=['importance'], ascending=False)
# imp.to_csv('importance.csv', index=False)

### Submit submission using terminal if current score better than parent holdout score
### then enter the score here for tracking
kagglescore = np.NaN

print("Submit to kaggle? : Y/N")

if input().lower() == 'y':
    df_test['item_cnt_month'] = np.expm1(pipe.predict(df_test.drop(columns=ignorecols)))

    submission = df_test.loc[: , ['ID', 'item_cnt_month']]

    submission['item_cnt_month'] = submission['item_cnt_month'].apply(lambda x: 0 if x<0 else (20 if x>20 else x))

    submission.to_csv(op+filename, index=False)
    print(f"submission file : {op+filename}")

    print(f"Run following command on terminal in submissions folder : ")
    print(f"kaggle competitions submit -c competitive-data-science-predict-future-sales -f {filename} -m '{flowargs['description']}'")

    print("Kaggle score press enter if you dont want to submit :")
    kagglescore = input()

    if kagglescore=='':
        print("Dont want to submit the current run? no issues we'll submit the next run")
    else:
        kagglescore = float(kagglescore)

    ### add multiple scores support to DeepFlow

else:
    print("We'll submit the next run")

flow.end_run()