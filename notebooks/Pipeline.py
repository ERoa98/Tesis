# libraries importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# additional modules
import sys
sys.path.append('../Algoritmos')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tsad.evaluating.evaluating import evaluating
from utils_functions import graficar_matriz_confusion, create_sequences

from LSTM_VAE import LSTM_VAE 
from sklearn.preprocessing import StandardScaler
import numpy as np

path_to_data = '../data/'

all_files=[]
import os
for root, dirs, files in os.walk(path_to_data):
    for file in files:
        if file.endswith(".csv"):
             all_files.append(os.path.join(root, file))

# datasets with anomalies loading
all_fault_data = [pd.read_csv(file, 
                          sep=';', 
                          index_col='datetime', 
                          parse_dates=True) for file in all_files if 'anomaly-free' not in file]
# anomaly-free df loading
anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], 
                            sep=';', 
                            index_col='datetime', 
                            parse_dates=True)

other_df = all_fault_data[0:15]
valve1_df = all_fault_data[15:31]
valve2_df = all_fault_data[31:36]

test_list = other_df[:2] + valve1_df[:2] + valve2_df[:2]




# Supongamos que 'data' es tu conjunto completo de datos en formato NumPy
# Dividir los datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2)

# Parámetros del modelo
input_dim = train_data.shape[-1]  # Número de características
timesteps = train_data.shape[1]   # Secuencias de tiempo
intermediate_dim = 64
latent_dim = 20
epochs = 50
validation_split = 0.2
BATCH_SIZE = 32
early_stopping = True

# Crear e inicializar el modelo LSTM VAE
model = LSTM_VAE(input_dim=input_dim, 
                 timesteps=timesteps, 
                 intermediate_dim=intermediate_dim, 
                 latent_dim=latent_dim, 
                 epochs=epochs, 
                 validation_split=validation_split, 
                 BATCH_SIZE=BATCH_SIZE, 
                 early_stopping=early_stopping)

# Entrenar el modelo
model.fit(train_data)

# Hacer predicciones en el conjunto de prueba
predicted = model.predict(test_data)

# Evaluar el modelo (aquí usamos Mean Squared Error como ejemplo)
mse = mean_squared_error(test_data.reshape(-1, input_dim), predicted.reshape(-1, input_dim))
print("Mean Squared Error en el conjunto de prueba:", mse)

# Aquí puedes agregar más análisis según tus necesidades