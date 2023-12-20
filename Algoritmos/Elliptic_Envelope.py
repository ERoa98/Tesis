from sklearn.covariance import EllipticEnvelope
import numpy as np

class Elliptic_Envelope:
    """
    Elliptic Envelope fits a robust covariance estimate to the data, and thus fits an ellipse to the central data points, ignoring points outside the central mode.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters: support_fraction, contamination, and random_state.
        
    Attributes
    ----------
    support_fraction : float
        The proportion of points to be included in the support of the raw MCD estimate.
    contamination : float
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    random_state : int
        The random seed used for reproducibility.
        
    Examples
    --------
    >>> from elliptic_envelope import Elliptic_Envelope
    >>> params = {'support_fraction': 0.9, 'contamination': 0.1, 'random_state': 42}
    >>> model = Elliptic_Envelope(params)
    >>> model.fit(X_train)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, params):
        self.support_fraction = params.get('support_fraction', None)
        self.contamination = params.get('contamination', 0.1)
        self.random_state = params.get('random_state', None)

    def _build_model(self):
        model = EllipticEnvelope(support_fraction=self.support_fraction,
                                 contamination=self.contamination,
                                 random_state=self.random_state)
        return model
    
    def fit(self, X):
        """
        Train the Elliptic Envelope model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        """
        self.model = self._build_model()
        self.model.fit(X)
    
    def predict(self, data):
        """
        Generate predictions using the trained Elliptic Envelope model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data, where -1 indicates an outlier and 1 indicates an inlier.
        """
        return self.model.predict(data)