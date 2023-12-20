from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import regularizers
import numpy as np

class LSTM_AE:
    """
    A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list of hyperparameters for the model, containing the following elements:
        - EPOCHS : int
            The number of training epochs.
        - BATCH_SIZE : int
            The batch size for training.
        - VAL_SPLIT : float
            The validation split ratio during training.

    Attributes
    ----------
    params : dict
        The hyperparameters for the model.

    Examples
    --------
    >>> from LSTM_AE import LSTM_AE
    >>> PARAMS = {EPOCHS: 100, BATCH_SIZE: 256, VAL_SPLIT: 0.1}
    >>> model = LSTM_AE(PARAMS)
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """
    
    def __init__(self, params):
        self.params = params
        self.threshold_factor = 15  # Factor multiplicativo para ajustar el umbral inicial
        self.shape = params['input_shape']  # Esto se establecerá en el método fit
        self.model = self._build_model()
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)

        

    def _build_model(self):
        self._Random(0)
        
        # Encoder
        inputs = Input(shape=(self.shape[1], self.shape[2]))
        encoded = LSTM(4, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.01))(inputs)
        # encoded = LSTM(5, activation='relu', return_sequences=True,
        #                kernel_regularizer=regularizers.l2(0.01))(encoded)
        # encoded = LSTM(10, activation='relu', return_sequences=True)(encoded)
        encoded = LSTM(2, activation='relu')(encoded)  # Última capa del encoder

        # Decoder
        decoded = RepeatVector(self.shape[1])(encoded)
        decoded = LSTM(2, activation='relu', return_sequences=True)(decoded)
        # decoded = LSTM(10, activation='relu', return_sequences=True)(decoded)
        # decoded = LSTM(5, activation='relu', return_sequences=True,
        #                kernel_regularizer=regularizers.l2(0.01))(decoded)
        decoded = LSTM(4, activation='relu', return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.01))(decoded)
        decoded = TimeDistributed(Dense(self.shape[2]))(decoded)

        model = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        model.compile(optimizer='adam', loss='mae', metrics=["mse"])
        
        return model
    
    def summary(self):
        """
        Imprime el resumen del modelo LSTM AE.
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("El modelo aún no se ha construido o inicializado.")
    
    def fit(self, X):
        """
        Train the sequence-to-sequence (LSTM-based) autoencoder model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        """

        self.shape = X.shape
        self.model = self._build_model()

        early_stopping = EarlyStopping(patience=5, 
                                       verbose=0)

        self.model.fit(X, X,
                  validation_split=self.params["VAL_SPLIT"],
                  epochs=self.params["EPOCHS"],
                  batch_size=self.params["BATCH_SIZE"],
                  verbose=0,
                  shuffle=False,
                  callbacks=[early_stopping]
                  )
    
    def predict(self, data):
        """
        Generate predictions using the trained sequence-to-sequence (LSTM-based) autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        recons = self.model.predict(data)
        return data, recons

    def update_threshold(self, X):
        """
        Calcula estadísticas móviles de los residuos y ajusta el umbral de detección de anomalías.
        """
        residuals = np.abs(X - self.model.predict(X))
        moving_average = np.mean(residuals, axis=1) # Calcula la media móvil de los residuos
        moving_std = np.std(residuals, axis=1) # Calcula la desviación estándar móvil de los residuos
        
        # Ajusta el umbral basándose en la desviación estándar móvil
        self.threshold = moving_average + self.threshold_factor * moving_std

    def detect_anomalies(self, X):
        """
        Detecta anomalías en los datos de entrada utilizando el umbral dinámico.
        """
        residuals = np.abs(X - self.model.predict(X))
        anomaly_mask = np.sum(residuals, axis=1) > self.threshold
        return anomaly_mask