import pandas as pd
import os

import goslate

os.listdir('data')

df_itemcategories = pd.read_csv('data/item_categories.csv')
df_shops = pd.read_csv('data/shops.csv')

df_itemcategories.head()
df_itemcategories.shape

df_shops.head()
df_shops.shape

gs = goslate.Goslate(service_urls=['http://translate.google.de'])  

df_shops['shop_name_en'] = [gs.translate(i, 'en') for i in df_shops['shop_name']]

df_itemcategories['item_category_name_en'] = [gs.translate(i, 'en') for i in df_itemcategories.item_category_name]

df_shops.to_csv('data/shops_en.csv')
df_itemcategories.to_csv('data/item_categories_en.csv', index=False)