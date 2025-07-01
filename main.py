import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

data_all = pd.read_csv("./data.csv")

data_all = data_all.dropna(axis=0, how='any')

data_without_na = pd.read_csv("./data_process.csv")
data_without_na = data_without_na.sort_values(['mmsi'], ascending=[True])

data_without_na = data_without_na.reset_index(drop=True)
ship_type = data_without_na['ship_type'].values.reshape(-1, 1)
enc = OneHotEncoder().fit(ship_type)
xxxx = enc.transform(ship_type).toarray() * data_without_na['ship_width'].values.reshape(-1, 1)
xxx = enc.transform(ship_type).toarray() * data_without_na['ship_length'].values.reshape(-1, 1)
#xxxx = enc.transform(ship_type).toarray()

#xx = data_without_na[['ship_length', 'ship_width']].values
xx = data_without_na['ship_length'].values.reshape(-1, 1) * data_without_na['ship_width'].values.reshape(-1, 1)
xx = enc.transform(ship_type).toarray() * xx.reshape(-1, 1)

lonx = data_without_na['lon'].values.reshape(-1, 1)
latx = data_without_na['lat'].values.reshape(-1, 1)
rotx = data_without_na['rot'].values.reshape(-1, 1)
sogx = data_without_na['sog'].values.reshape(-1, 1)

#, pd.DataFrame(lonx), pd.DataFrame(latx)  , pd.DataFrame(rotx), pd.DataFrame(sogx)  , data_without_na.iloc[:, -10:]
X = pd.concat([pd.DataFrame(xx), pd.DataFrame(xxx), pd.DataFrame(xxxx), data_without_na[['lon', 'lat', 'rot', 'sog']], data_without_na.iloc[:, -10:]], axis=1)
y = data_without_na[['draught']]
#X, y = shuffle(X, y, random_state=0)

#linearModel = DecisionTreeRegressor(criterion="mae", max_depth=15, min_samples_leaf=10)
linearModel = DecisionTreeRegressor(criterion="mse", max_depth=3, min_samples_leaf=5000)
#, max_depth=12, min_samples_leaf=10
#linearModel = LinearRegression()

scores = cross_validate(linearModel, X=X, y=y, cv=2, return_train_score=True, return_estimator=True,
                        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error', 'r2'])

#regulazation:
print('ship_length', 'ship_width', 'cluster', ' rot', 'sog lon lat', 'filter: ship_type', 'time series model')
print('test MAE Scores:', -1 * scores['test_neg_mean_absolute_error'])

print('test MSE Scores:', -1 * scores['test_neg_mean_squared_error'])

print('test ME Scores:', -1 * scores['test_max_error'])

print('train Scores:', -1 * scores['train_neg_mean_absolute_error'])

model_list = scores['estimator']


"""
Base models: linear model, tree model
Different features: locations, speeds, turning rates


#
# data_without_na = data_all[['ship_length', 'ship_width', 'draught']]
# X = data_without_na[['ship_length', 'ship_width']]
# y = data_without_na[['draught']]
# X, y = shuffle(X, y, random_state=0)
#
# linearModel = LinearRegression()
#
# scores = cross_validate(linearModel, X=X, y=y, cv=2, return_train_score=True, return_estimator=True,
#                         scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error', 'r2'])
#
# print('ship_length', 'ship_width')
# print('test MAE Scores:', -1 * scores['test_neg_mean_absolute_error'])
#
# print('test MSE Scores:', -1 * scores['test_neg_mean_squared_error'])
#
# print('test ME Scores:', -1 * scores['test_max_error'])
#
# print('train Scores:', -1 * scores['train_neg_mean_absolute_error'])
#
# model_list = scores['estimator']

#
# data_without_na = data_all[['ship_length', 'ship_width', 'draught', 'ship_type']]
# ship_type = data_without_na['ship_type'].values.reshape(-1, 1)
# enc = OneHotEncoder().fit(ship_type)
# xxx = enc.transform(ship_type).toarray()
# xx = data_without_na[['ship_length', 'ship_width']].values
#
# X = pd.concat([pd.DataFrame(xx), pd.DataFrame(xxx)], axis=1)
# y = data_without_na[['draught']]
# X, y = shuffle(X, y, random_state=0)
#
# linearModel = LinearRegression()
#
# scores = cross_validate(linearModel, X=X, y=y, cv=2, return_train_score=True, return_estimator=True,
#                         scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error', 'r2'])
#
# print('ship_length', 'ship_width', 'ship_type')
# print('test MAE Scores:', -1 * scores['test_neg_mean_absolute_error'])
#
# print('test MSE Scores:', -1 * scores['test_neg_mean_squared_error'])
#
# print('test ME Scores:', -1 * scores['test_max_error'])
#
# print('train Scores:', -1 * scores['train_neg_mean_absolute_error'])
#
# model_list = scores['estimator']

#
# data_without_na = data_all[['ship_length', 'ship_width', 'draught', 'ship_type']]
# ship_type = data_without_na['ship_type'].values.reshape(-1, 1)
# enc = OneHotEncoder().fit(ship_type)
# xxx = enc.transform(ship_type).toarray()
# xx = data_without_na[['ship_length', 'ship_width']].values
#
# X = pd.concat([pd.DataFrame(xx), pd.DataFrame(xxx)], axis=1)
# y = data_without_na[['draught']]
# X, y = shuffle(X, y, random_state=0)
#
# linearModel = DecisionTreeRegressor()
#
# scores = cross_validate(linearModel, X=X, y=y, cv=2, return_train_score=True, return_estimator=True,
#                         scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error', 'r2'])
#
# print('ship_length', 'ship_width', 'ship_type')
# print('test MAE Scores:', -1 * scores['test_neg_mean_absolute_error'])
#
# print('test MSE Scores:', -1 * scores['test_neg_mean_squared_error'])
#
# print('test ME Scores:', -1 * scores['test_max_error'])
#
# print('train Scores:', -1 * scores['train_neg_mean_absolute_error'])
#
# model_list = scores['estimator']


data_without_na = data_all[['mmsi', 'time_id', 'ship_length', 'ship_width', 'draught', 'ship_type', 'lon', 'lat', 'rot', 'sog']]
df_mask = data_without_na["ship_type"].isin(["dredging", "tanker", "passenger", "hsc", "cargo", "wig"])

data_without_na = data_without_na[df_mask]


# time series
df_sorted = data_without_na.sort_values(['mmsi', 'time_id'], ascending=[True, True])
grouped = df_sorted.groupby('mmsi', sort=False)   #.head(indx)
dfs = []
for name, group in grouped:
    dfs.append(group)

for i, df_small in enumerate(dfs):
    pre = df_small
    pre = pre.reset_index(drop=True)
    for history in range(1, 11):
        cur = df_small[0:1]
        df_small = pd.concat([df_small[0:1], df_small],  axis=0, ignore_index=True)
        df_small = df_small.iloc[:-1]
        cur = pd.DataFrame(df_small['draught'].values.reshape(-1, 1), columns=['draught'+str(history)])
        pre = pd.concat([pre, cur],  axis=1)
    dfs[i] = pre

data_without_na = pd.concat(dfs,  axis=0, ignore_index=True)
data_without_na = data_without_na.sort_values(['time_id'], ascending=[True])
data_without_na.to_csv("./data_process.csv")
"""

