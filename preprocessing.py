import numpy as np
from helpers import load_csv_data
r"""
data_path = (
    r"C:\Users\lilar\Documents\Ma1\machine learning\MLproject1\dataset_to_release\dataset"
)
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
np.savez_compressed("data_loaded.npz", x_train=x_train, y_train=y_train, x_test=x_test, train_ids=train_ids, test_ids=test_ids)
"""


#to access to the data without running again --> 

data = np.load("data_loaded.npz")
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
train_ids = data["train_ids"]
test_ids = data["test_ids"]


def explore_dimensions(y_train, x_train, x_test=None):
    """Compute and display dataset dimensions, shapes, and class distribution.

    Args:
        y_train: shape=(N_train, 1) or (N_train,)
        x_train: shape=(N_train, D)
        x_test: shape=(N_test, D), optional

    Returns:
        N_train: number of training samples (int), number of rows
        D:number of features (int), number of columns
    """
    assert (y_train.shape[0] == x_train.shape[0]), "y_train and x_train must have the same number of rows"
    if x_test is not None:
        assert (x_train.shape[1] == x_test.shape[1]), "x_train and x_test must have the same number of features"

    N_train, D = x_train.shape
    classes, counts = np.unique(y_train, return_counts=True)

    print(f"Training samples  (N_train): {N_train}")
    print(f"Features          (D)      : {D}")
    print(f"Classes: {classes.tolist()} | Counts: {counts.tolist()}")

    if x_test is not None:
        N_test = x_test.shape[0]
        print(f"Test samples      (N_test) : {N_test}")

    return N_train, D

N, D = explore_dimensions(y_train, x_train, x_test)