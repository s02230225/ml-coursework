import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    lst = list()
    t = num_objects // num_folds
    ind = np.arange(0, num_objects, dtype=int)

    # Первые num_folds - 1 разделим поровну. Последнее деление бахнем обособленно
    for i in range(num_folds-1):
        temp_arr = ind[i*t: (i+1) * t]
        lst.append((np.delete(ind, temp_arr), temp_arr))

    temp_arr = ind[(num_folds-1) * t:]
    lst.append((np.delete(ind, temp_arr), temp_arr))

    return lst


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """

    folds_num = len(folds)
    ans = dict()

    for sclr_tuple in parameters['normalizers']:
        sclr = sclr_tuple[0]  # Taking func from tuple
        sclr_name = sclr_tuple[1]  # Taking name

        for n_num in parameters['n_neighbors']:

            for t_metr in parameters['metrics']:

                for w in parameters['weights']:

                    value = 0
                    for fold in folds:
                        y_train = y[fold[0]]
                        y_test = y[fold[1]]

                        if sclr is None:
                            X_train = X[fold[0]]
                            X_test = X[fold[1]]
                        else:
                            X_train = sclr.fit_transform(X[fold[0]])
                            X_test = sclr.transform(X[fold[1]])

                        clf = knn_class(n_neighbors=n_num,
                                        metric=t_metr, weights=w)  # Making knn class with parameters
                        clf.fit(X_train, y_train)  # Train knn class

                        # Predicting with trained model
                        y_pred = clf.predict(X_test)
                        value += score_function(y_test, y_pred)

                    value = value / folds_num  # Making mean
                    ans[(sclr_name, n_num, t_metr, w)
                        ] = value  # Adding to dict

    print(ans)
    return ans
