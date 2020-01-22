import numpy as np
import os
from sklearn.model_selection import train_test_split


# list all the files in the input dir
def read_files():
    for dirname, _, filenames in os.walk('../input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def load_data_and_fix_map(X, Y):
    # print(X.shape[0], X.shape[1])
    # print(y.shape[0], y.shape[1])
    # print(X[0].shape)
    Y = np.zeros(X.shape[0])
    Y[:204] = 9
    Y[204:409] = 0
    Y[409:615] = 7
    Y[615:822] = 6
    Y[822:1028] = 1
    Y[1028:1236] = 8
    Y[1236:1443] = 4
    Y[1443:1649] = 3
    Y[1649:1855] = 2
    Y[1855:] = 5
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=.02,
                                                        random_state=2)
    print(f"shape of the train features {x_train.shape}")
    print(f"shape of the train target {y_train.shape}")


def main():
    X = np.load('../input\X.npy')
    y = np.load('../input\Y.npy')
    read_files()
    load_data_and_fix_map(X, y)


if __name__ == "__main__":
    main()
