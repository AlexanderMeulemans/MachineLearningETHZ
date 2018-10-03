from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

class OutlierExtractor(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """
        try:
            self.threshold = kwargs.pop('neg_conf_val')
        except KeyError:
            self.threshold = -10.0
        pass
        self.kwargs = kwargs

    def transform(self, X,Y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        x = np.asarray(X)
        y = np.asarray(Y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return x[lcf.negative_outlier_factor_ > self.threshold, :], y[lcf.negative_outlier_factor_ > self.threshold]

    def fit(self, *args, **kwargs):
        return self

    def get_params(self,deep = True):
        if deep:
            return {'threshold': self.threshold}
        else:
            return {'threshold': self.threshold}

    def set_params(self,**kwargs):
        self.threshold = kwargs['threshold']
        return self

class AdvancedImputer(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to impute missing values by regression or nearest neighbour
        Keyword Args:
            regressor(string): either linear or nearest_neighbour. Decides which method to use

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """
        try:
            self.threshold = kwargs.pop('neg_conf_val')
        except KeyError:
            self.threshold = -10.0
        pass
        self.kwargs = kwargs

    def transform(self, X,Y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        x = np.asarray(X)
        y = np.asarray(Y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return x[lcf.negative_outlier_factor_ > self.threshold, :], y[lcf.negative_outlier_factor_ > self.threshold]

    def fit(self, *args, **kwargs):
        return self