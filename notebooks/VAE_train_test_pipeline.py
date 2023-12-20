#%%
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

#%%
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

N_STEPS = 40
Q = 0.9 # quantile for upper control limit (UCL) selection
input_dim = 8  # Suponiendo que cada punto de datos en tus series temporales tiene 30 características
timesteps = 40  # Suponiendo que estás utilizando secuencias de 10 pasos de tiempo
intermediate_dim = 16
latent_dim = 4
epochs = 100
validation_split = 0.2
BATCH_SIZE = 1
early_stopping = True

# Crear la instancia de LSTM_VAE con los parámetros deseados
model = LSTM_VAE(input_dim=input_dim, 
                 timesteps=timesteps, 
                 intermediate_dim=intermediate_dim, 
                 latent_dim=latent_dim, 
                 epochs=epochs, 
                 validation_split=validation_split, 
                 BATCH_SIZE=BATCH_SIZE, 
                 early_stopping=early_stopping)
# model = LSTM_VAE()
#%%
predicted_outlier, predicted_cp = [], []
for df in test_list:
    X_train = df[:400].drop(['anomaly','changepoint'], axis=1)
    
    # scaler init and fitting
    StSc = StandardScaler()
    StSc.fit(X_train)
    
    # convert into input/output
    X = create_sequences(StSc.transform(X_train), N_STEPS)
    
    # model fitting
    # history, model = arch(X)
    model.fit(X)
    
    # results predicting
    residuals = pd.Series(np.sum(np.mean(np.abs(X - model.predict(X)[1]), axis=1), axis=1))
    UCL = residuals.quantile(Q) * 3/2
    
    # results predicting
    X = create_sequences(StSc.transform(df.drop(['anomaly','changepoint'], axis=1)), N_STEPS)
    cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X - model.predict(X)[1]), axis=1), axis=1))
    
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > UCL
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
    
    prediction = pd.Series(data=0, index=df.index)
    prediction.iloc[anomalous_data_indices] = 1
    
    # predicted outliers saving
    predicted_outlier.append(prediction)
    
    # predicted CPs saving
    prediction_cp = abs(prediction.diff())
    prediction_cp[0] = prediction[0]
    predicted_cp.append(prediction_cp)

#%%
# Trazar outliers
# true outlier indices selection
true_outlier = [df.anomaly for df in test_list]

plt.figure(figsize=(12,3))  # Crea una nueva figura para los outliers
predicted_outlier[0].plot(label='predictions', marker='o', markersize=5)
true_outlier[0].plot(marker='o', markersize=2)
plt.legend()
plt.title('Outliers Comparison')
plt.show()  # Muestra la figura de outliers

# Trazar changepoints

true_cp = [df.changepoint for df in test_list]

plt.figure(figsize=(12,3))  # Crea una nueva figura para los changepoints
predicted_cp[0].plot(label='predictions', marker='o', markersize=5)
true_cp[0].plot(marker='o', markersize=2)
plt.legend()
plt.title('Changepoints Comparison')
plt.show()  # Muestra la figura de changepoints


# binary classification metrics calculation
binary = evaluating(
    true_outlier, 
    predicted_outlier, 
    metric='binary'
)

conf_matrix = evaluating(
    true_outlier, 
    predicted_outlier, 
    metric='confusion_matrix'
)
graficar_matriz_confusion(conf_matrix)


# average detection delay metric calculation
add = evaluating(
    true_cp, 
    predicted_cp, 
    metric='average_time',
    anomaly_window_destenation='righter', 
    portion=1
)

# nab metric calculation
nab = evaluating(
    true_cp, 
    predicted_cp, 
    metric='nab', 
    numenta_time='30S',
    anomaly_window_destenation='center', 
)

# %%
