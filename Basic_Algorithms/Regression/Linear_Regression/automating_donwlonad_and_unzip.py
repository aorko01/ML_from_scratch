import os
import tarfile
from six.moves import urllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

Download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
# Make Housing_path relative to the script location
Housing_path = os.path.join(SCRIPT_DIR, "datasets/housing")
Housing_url = Download_root + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=Housing_url, housing_path=Housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()  


if __name__ == "__main__":
    fetch_housing_data()
