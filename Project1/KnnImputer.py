from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import fancyimpute as fi

class KnnImputer(TransformerMixin):
    def __init__(self, missing_values=np.nan):
        """
        Impute missing values to k nearest neighbour, class to implement this in pipeline.
        """
        self.knn = fi.KNN()

    def fit_transform(self, X, y=None):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        x = np.asarray(X)
        x, mask = self.knn.prepare_input_data(x)
        x_res = self.knn.solve(x, mask)
        return x_res

