### define params
params = {
    'Pipeline' : "make_pipeline(OneHotEncoder(handle_unknown='ignore'), LGBMRegressor(n_jobs=-1))",
    'ip' : '../data',
    'op' : '../submissions/',
    'seed' : 123,
    'trainstart' : '201401',
    'holdstart' : '201506',
    'folds' : 5, ### No of holdout runs to trigger
    'holdoutmonths' : 1, ### No of months per holdout
    'holdoutshift' : 1, ### Shift holdout by months
    'laglist' : "list(range(1, 13))",
    'mkey_cols' : ['shop_id', 'item_id'],
    'categoricalcols' : ['shop_id', 'item_id', 'Year', 'Month', 'item_category_id'],
    'numericcols' : ['item_price', 'item_cnt_day'],
    'ignorecols' : ['ID', 'item_cnt_day', 'period','item_price', 'item_id'],
    'targetcol' : 'item_cnt_day'
}


flowargs = {
    'projectname' : 'Kaggle - predict future sales',
    'runmasterfile' : '../runmaster.csv',
    'description' : 'Passing pipeline as string - and submitting directly to kaggle',
    'benchmark' : 1,
    'parentID' : 27,
    'params' : params
}
