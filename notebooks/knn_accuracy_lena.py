import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



def knn_accuracy(embeddings, true_labels, test_size=0.1, k = 10, rs=42, metric="euclidean"):
    """Calculates kNN accuracy.
    
    Parameters
    ----------
    embeddings : list 
        List with the different datasets for which to calculate the kNN accuracy.
    true_labels : array-like
        Array with labels (colors).
    k : int, default=10
        Number of nearest neighbors to use.
    rs : int, default=42
        Random seed.
    metric : str, default="euclidean"
        Metric to use for the distances computation (e.g. "euclidean", "cosine", etc.).
    
    Returns
    -------
    knn_accuracy : float
        kNN accuracy of the dataset.
    
    """

    X_train, X_test, y_train, y_test = train_test_split(embeddings, true_labels, test_size=test_size, random_state = rs)
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
    knn = knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)

    
    return knn_accuracy