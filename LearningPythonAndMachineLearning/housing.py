import os
import tarfile
from typing import final
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
DOWLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWLOAD_ROOT +HOUSING_PATH +"/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


def split_data_test(data, test_ratio):
    
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(indentifier, test_ratio, hash):
    return hash(np.int64(indentifier)).digest()[-1] < 256 *test_ratio

def split_data_test_by_id(data, test_ratio,id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()
housing_with_id = housing.reset_index()
train_set, test_set = split_data_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_test_by_id(housing_with_id, 0.2,"id")
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(housing["income_cat"].value_counts()/len(housing))
for set in (strat_train_set,strat_test_set):
    set.drop(["income_cat"],axis = 1, inplace = True)


################################
housing = strat_train_set.copy()
housing.plot(kind = "scatter", x ="longitude", y = "latitude",alpha = 0.4, s = housing["population"]/100, label = "population",
c = "median_house_value",cmap = plt.get_cmap("jet"),colorbar = True)
# plt.legend()
# plt.show()

corr_matrix  = housing.corr()

# print(corr_matrix["median_house_value"].sort_values(ascending = False))

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize = (12, 8))

housing.plot(kind = "scatter",x = "median_income",y = "median_house_value",alpha = 0.1)
# plt.show()
# train_set, test_set = split_data_test(housing, 0.2)
# print(len(train_set),"train +", len(test_set),"set")
#print(housing.head())
#housing.info()
#print(housing["ocean_proximity"].value_counts())
#print(housing.describe())
#housing.hist(bins = 50, figsize=(20,15))
#plt.show()
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending = False))
housing = strat_train_set.drop("median_house_value",axis = 1)
housing_label = strat_train_set["median_house_value"].copy()

housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) # option 3
imputer = SimpleImputer(strategy = "median")

housing_num = housing.drop("ocean_proximity", axis = 1)
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
x = imputer.transform(housing_num)
housing_str = pd.DataFrame(x, columns = housing_num.columns)

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]

housing_cat_encoded = encoder.fit_transform(housing_cat)

print(housing_cat_encoded)
print(encoder.classes_)

########

encoder = OneHotEncoder()

housing_cat_1Hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1Hot.toarray())
################################################################
encoder = LabelBinarizer()

housing_cat_1Hot = encoder.fit_transform(housing_cat)
print(housing_cat_1Hot)
 ###############################################################
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = True):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)

housing_extra_attribs = attr_adder.transform(housing.values)
np_pineline = Pipeline([('imputer', SimpleImputer(strategy = "median")),('attribs_adder',CombinedAttributesAdder()),('std_scaler',StandardScaler())])
housing_num_tr = np_pineline.fit_transform(housing_num)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
class DataFrameCollector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# np_pipeline = Pipeline([("selector",DataFrameCollector(num_attribs)),
#                         ('imputer', SimpleImputer(strategy = "median")),
#                         ('attribs_adder',CombinedAttributesAdder()),
#                         ('std_scaler',StandardScaler()),
#                         ])

# cap_pipeline = Pipeline([("selector",DataFrameCollector(cat_attribs))
#                             ,('label_binarizer',LabelBinarizer())])

full_pipeline = ColumnTransformer([("num",np_pineline,num_attribs),
                                    ("cat",OneHotEncoder(),cat_attribs),])
housing_prepare = full_pipeline.fit_transform(housing)
print(housing_prepare.shape)

lin_neg = LinearRegression()
lin_neg.fit(housing_prepare, housing_label)

some_data= housing.iloc[5]
some_label = housing_label.iloc[5]
# some_data_prepare = full_pipeline.transform(some_data)
# print("Prediction:   ", lin_neg.predict(some_data_prepare))
# print("Labels: ", some_label)
from sklearn.metrics import mean_squared_error
housing_predictions = lin_neg.predict(housing_prepare)
lin_mse = mean_squared_error(housing_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepare, housing_label)

housing_predictions = tree_reg.predict(housing_prepare)
tree_mse = mean_squared_error(housing_label,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepare, housing_label, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores: ",scores)
    print("Mean: ", scores.mean())
    print("Standard deviation:", scores.std())

# //display_scores(rmse_scores)

lin_scores = cross_val_score(lin_neg, housing_prepare, housing_label, scoring= "neg_mean_squared_error", cv = 10)

lin_rsme_scores = np.sqrt(-lin_scores)
display_scores(lin_rsme_scores)

from sklearn.ensemble import RandomForestRegressor

# forest_reg = RandomForestRegressor()

# forest_reg.fit(housing_prepare, housing_label)

# housing_predictions = forest_reg.predict(housing_prepare)

# forest_mse = mean_squared_error(housing_label, housing_predictions)

# forest_rmse = np.sqrt(forest_mse)

# print(forest_rmse)

# forest_scores = cross_val_score(forest_reg, housing_prepare, housing_label, scoring = "neg_mean_squared_error", cv = 10)

# forest_rmse_scores = np.sqrt(-forest_scores)

# display_scores(forest_rmse_scores)

from sklearn.model_selection import GridSearchCV
# Cau 1
from sklearn.svm import SVR
# param_grid = [
#     {'kernel':['linear'],'C':[10., 30., 100., 300., 1000., 3000., 10000., 30000.0,]},
#     {'kernel':['rbf'],'C':[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
#     'gamma': [0.01, 0.03,0.1,0.3,1.0,3.0]}
# ]
# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error',verbose = 2)
# print(grid_search.fit(housing_prepare, housing_label))
# nav_reg = grid_search.best_score_
# rmse = np.sqrt(-nav_reg)
# print(rmse)
# print(grid_search.best_params_)
#Cau 2
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, expon
param_grid = [
    {'kernel': ['linear','rbf'],
    'C': reciprocal(20, 200000),
    'gamma': expon(scale = 1.0)
    }
]
svm_reg = SVR()
rnd_reg = RandomizedSearchCV(svm_reg, param_distributions=param_grid, n_iter= 50, cv = 5, scoring = 'neg_mean_squared_error', verbose = 2, random_state = 42)
# print(rnd_reg.fit(housing_prepare, housing_label))
# mse = rnd_reg.best_score_
# rmse = np.sqrt(-mse)
# print(rmse)
# print(rnd_reg.best_params_)

# expon_distrib = expon(scale= 1.)
# samples = expon_distrib.rvs(10000, random_state= 42)
# plt.figure(figsize= (10, 4))
# plt.subplot(121)
# plt.title('Exponential distribution (Scale= 1.0)')
# plt.hist(samples, bins = 50)
# plt.subplot(122)
# plt.title('Log of this distribution')
# plt.hist(np.log(samples), bins = 50)
# plt.show()


# reciprocal_distribution = reciprocal(20, 20000)
# samples = reciprocal_distribution.rvs(10000, random_state= 42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title('Reciprocal distribution(Scale = 1.0)')
# plt.hist(samples, bins = 50)
# plt.subplot(122)
# plt.title('Log this distribution')
# plt.hist(np.log(samples), bins = 50)
# plt.show()
# #Cau 3

def indeces_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr),-k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices = indeces_of_top_k(self.feature_importances,self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices]

k = 5
# param_grid = [
#  {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#  ]
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
# scoring='neg_mean_squared_error', return_train_score = True)

# grid_search.fit(housing_prepare,housing_label)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

# cvres =grid_search.cv_results_

# for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
#     print(np.sqrt((-mean_score)),params)

# feature_importances = grid_search.best_estimator_.feature_importances_

# print(feature_importances)

# extra_attribs = ["rooms_per_hhold", "pop_per_household","bedrooms_per_room"]
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# for i in sorted(zip(feature_importances, attributes),reverse=True):
#     print(i)


# #test_set

# final_model = grid_search.best_estimator_

# X_test = strat_test_set.drop("median_house_value", axis = 1)

# y_test = strat_test_set["median_house_value"].copy()

# X_test_prepared = full_pipeline.transform(X_test)

# final_predictions = final_model.predict(X_test_prepared)

# final_mse = mean_squared_error(y_test, final_predictions)

# final_rmse = np.sqrt(final_mse) 
# print(final_rmse)

# from scipy import stats

# confidence = 0.95

# squared_error = (final_predictions - y_test)**2

# print(np.sqrt(stats.t.interval(confidence, len(squared_error) - 1,loc = squared_error.mean(),scale = stats.sem(squared_error))))