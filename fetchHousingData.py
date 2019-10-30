import os
import tarfile
from six.moves import urllib
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1 TODO: replace gaps in data by median
# 2 TODO: replace strings in dataframe by vectors ex. 000100

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("zestawy danych", "mieszkania")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
TEST_RATIO = 0.2


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    print("Data downloaded.")
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    data = pd.read_csv(csv_path)
    # imputer = Imputer(strategy="median") fill gaps in table
    return data


def load_train_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "train.csv")
    return pd.read_csv(csv_path)


def load_test_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "test.csv")
    return pd.read_csv(csv_path)


def split_train_test(dataset, test_ratio):
    data = dataset.copy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    test_data = None
    train_data = None
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)
    for train_index, test_index in split.split(data, data["income_cat"]):
        test_data = data.loc[test_index]
        train_data = data.loc[train_index]
    for set in (train_data, test_data):
        set.drop("income_cat", axis=1, inplace=True)
    # print(train_data)
    # print(test_data)
    # shuffled_indices = np.random.permutation(len(data))
    # test_set_size = int(len(data) * test_ratio)
    # test_indicies = shuffled_indices[:test_set_size]
    # train_indicies = shuffled_indices[test_set_size:]
    # train_data = data.iloc[train_indicies]
    # test_data = data.iloc[test_indicies]

    train_data.to_csv(os.path.join(HOUSING_PATH, "train.csv"), index=None, header=True)
    test_data.to_csv(os.path.join(HOUSING_PATH, "test.csv"), index=None, header=True)
    print("Train and Test data saved to .csv files")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    roomsId, bedroomsId, papulationId, householdId = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_family = X[:, self.roomsId] / X[:, self.householdId]
        population_per_family = X[:, self.papulationId] / X[:, self.householdId]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedroomsId] / X[:, self.roomsId]
            return np.c_[X, rooms_per_family, population_per_family, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_family, population_per_family]


num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


def encoded_data_without_gaps():
    data = load_train_data()
    imputer = Imputer("median")
    housing_num = data.drop("ocean_proximity", axis=1)  # usuwamy kolumne ze stringami
    # housing_num_tr = num_pipeline.fit_tranfrom(housing_num)
    imputer.fill(housing_num)
    X = imputer.transform(housing_num)
    housing_wihout_gaps = pd.DataFrame(X, columns=housing_num.columns)
    encoder = LabelEncoder()
    housing_cat = data["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)

    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1)).toarray()
    housing_wihout_gaps["ocean_proximity"] = housing_cat_1hot
    return housing_wihout_gaps


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transfrom(self, X):
        return X[self.attribute_names].values


# przygotowanie danych - wypelnianie mediana pustych danych numerycznych
# a stringi zastapienie wektorami normalnymi
# zwraca macierz numpy
def dataprepare(data):
    cat_attribs = ["ocean_proximity"]
    num_attribs = list(data.drop(cat_attribs, axis=1))

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding='onehot-dense')),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    return full_pipeline.fit_transform(data)


def trainRegresion(data):
    lin_reg = LinearRegression()
    housing_prepared = dataprepare(load_train_data())
    lin_reg.fit(housing_prepared, housing_labels)


def datawork(data):
    # s - dlugosc kolka, # c - co oznaczyc kolorami
    # cmap - mama (tutaj domyslna) o nazwie jet
    # data.plot(kind="scatter", x="longitude",  y="latitude", alpha=0.4,
    #    s=data["population"]/100, label="Populacja", figsize=(10, 7),
    #    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    # )
    # wyliczanie macierzy korelacji, dla kazdej mozliwej pary
    # corr_matrix = data.corr()
    # stopien korelacji kazdego atrybutu z mediana cen mieszkan
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))
    # plt.legend()
    # plt.show()
    # housing = load_train_data().drop("median_house_value", axis=1)
    housing_labels = load_train_data()["median_house_value"].copy()
    print(housing_labels)
    print("ss")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("usage:", sys.argv[0], "[download/show/split/work]")
    elif sys.argv[1] == "download":
        fetch_housing_data()
    elif sys.argv[1] == "show":
        data = load_housing_data()
        for k in data:
            print(k)
        # print(data["ocean_proximity"].value_counts())
        # print(data.describe())
        # data.hist(bins=50, figsize=(20, 15))
        # plt.show()
    elif sys.argv[1] == "split":
        split_train_test(load_housing_data(), TEST_RATIO)
    elif sys.argv[1] == "work":
        datawork(load_train_data())
    else:
        print("usage:", sys.argv[0], "[download/show/split/work]")
