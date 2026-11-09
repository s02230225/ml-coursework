import numpy as np
from collections import Counter


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    if len(sample) == 0:
        return {'gini': 0.0, 'entropy': 0.0, 'error': 0.0}

    # Count class frequencies
    class_counts = Counter(sample)
    total = len(sample)
    probabilities = np.array([count / total for count in class_counts.values()])

    # Calculate Gini impurity
    gini = 1 - np.sum(probabilities ** 2)

    # Calculate entropy (using natural logarithm)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # adding small epsilon to avoid log(0)

    # Calculate classification error
    max_class_count = max(class_counts.values())
    error = 1 - (max_class_count / total)

    measures = {
        'gini': float(gini),
        'entropy': float(entropy),
        'error': float(error)
    }
    return measures
