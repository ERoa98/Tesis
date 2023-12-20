from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class LSTM_VAE:
    """
    A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    TenserFlow_backend : bool, optional
        Flag to specify whether to use TensorFlow backend (default is False).

    Attributes
    ----------
    None
        
    Examples
    -------
    >>> from LSTM_VAE import LSTM_VAE
    >>> model = LSTM_VAE()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """
    
    def __init__(self, 
                 input_dim=None, 
                 timesteps=None, 
                 intermediate_dim=32, 
                 latent_dim=100, 
                 epochs=20, 
                 validation_split=0.1, 
                 BATCH_SIZE=1, 
                 early_stopping=True): 
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = BATCH_SIZE
        self.early_stopping = early_stopping
        self.epsilon_std = 1.
        self.model = None
        self.enc = None
        self.gen = None
        
    def _build_model(self, 
                     input_dim, 
                     timesteps, 
                     intermediate_dim, 
                     latent_dim):
        
        self._Random(0)

        x = Input(shape=(timesteps, input_dim,))

        h = LSTM(intermediate_dim)(x)

        self.z_mean = Dense(latent_dim)(h)
        self.z_log_sigma = Dense(latent_dim)(h)
        
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_sigma]) 

        decoder_h = LSTM(intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(input_dim, return_sequences=True)

        h_decoded = RepeatVector(timesteps)(z)
        h_decoded = decoder_h(h_decoded)

        x_decoded_mean = decoder_mean(h_decoded)

        vae = Model(x, x_decoded_mean)

        encoder = Model(x, self.z_mean)

        decoder_input = Input(shape=(latent_dim,))

        _h_decoded = RepeatVector(timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)

        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        vae.compile(optimizer='rmsprop', loss=self.vae_loss)

        return vae, encoder, generator
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def sampling(self, args):
        """
        Sample from the latent space using the reparameterization trick.

        Parameters
        ----------
        args : list
            List of tensors [z_mean, z_log_sigma].

        Returns
        -------
        z : tensorflow.Tensor
            Sampled point in the latent space.
        """
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                  mean=0., stddev=self.epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    def vae_loss(self, x, x_decoded_mean):
        """
        Calculate the VAE loss.

        Parameters
        ----------
        x : tensorflow.Tensor
            Input data.
        x_decoded_mean : tensorflow.Tensor
            Decoded output data.

        Returns
        -------
        loss : tensorflow.Tensor
            VAE loss value.
        """
        mse = losses.MeanSquaredError()
        xent_loss = mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    def fit(self, data):
        """
        Train the LSTM variational autoencoder model on the provided data using instance attributes.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training.
        """
        self.model, self.enc, self.gen = self._build_model(self.input_dim, 
        timesteps=self.timesteps, 
        intermediate_dim=self.intermediate_dim,
        latent_dim=self.latent_dim)


        # Preparar callbacks para el entrenamiento
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=1))

        # Entrenamiento del modelo
        self.model.fit(data,
                       data,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=self.validation_split,
                       callbacks=callbacks,
                       verbose=1)
    
    def predict(self, data):
        """
        Generate predictions using the trained LSTM variational autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for making predictions.

        Returns
        -------
        predictions : numpy.ndarray
            The reconstructed output predictions.
        """
        
        return self.model.predict(data)

    