import os
import tarfile
from six.moves import urllib
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

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
    return pd.read_csv(csv_path)

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
    #print(train_data)
    #print(test_data)
    #shuffled_indices = np.random.permutation(len(data))
    #test_set_size = int(len(data) * test_ratio)
    #test_indicies = shuffled_indices[:test_set_size]
    #train_indicies = shuffled_indices[test_set_size:]
    #train_data = data.iloc[train_indicies]
    #test_data = data.iloc[test_indicies]
    train_data.to_csv(os.path.join(HOUSING_PATH, "train.csv"), index=None, header=True)
    test_data.to_csv(os.path.join(HOUSING_PATH, "test.csv"), index=None, header=True)
    print("Train and Test data saved to .csv files")

def datawork(data):
    # s - dlugosc kolka, # c - co oznaczyc kolorami
    # cmap - mama (tutaj domyslna) o nazwie jet
    data.plot(kind="scatter", x="longitude",  y="latitude", alpha=0.4,
        s=data["population"]/100, label="Populacja", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    )
    # wyliczanie macierzy korelacji, dla kazdej mozliwej pary
    corr_matrix = data.corr()
    # stopien korelacji kazdego atrybutu z mediana cen mieszkan
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("usage:", sys.argv[0], "[download/show/split/work]")
    elif sys.argv[1] == "download":
        fetch_housing_data()
    elif sys.argv[1] == "show":
        data = load_housing_data()
        print(data)
        #print(data["ocean_proximity"].value_counts())
        #print(data.describe())
        #data.hist(bins=50, figsize=(20, 15))
        #plt.show()
    elif sys.argv[1] == "split":
        split_train_test(load_housing_data(), TEST_RATIO)
    elif sys.argv[1] == "work":
        datawork(load_train_data())
    else:
        print("usage:", sys.argv[0], "[download/show/split/work]")
    