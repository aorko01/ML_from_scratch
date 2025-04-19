import pandas as pd
import os
from automating_donwlonad_and_unzip import fetch_housing_data


def load_housing_data(housing_path="datasets/housing"):
    fetch_housing_data()
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    housing = load_housing_data()
    print(housing.head())
    print(housing.info())
