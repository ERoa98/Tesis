import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def graficar_matriz_confusion(confusion_tuple):
    """
    Grafica la matriz de confusión usando Seaborn.

    Parameters:
    confusion_tuple (tuple): Tupla conteniendo TP, TN, FP, FN.
    """
    # Desempaquetar la tupla
    TP, TN, FP, FN = confusion_tuple

    # Crear la matriz de confusión
    matriz_confusion = np.array([[TP, FN],
                                 [FP, TN]])

    # Usar Seaborn para crear un mapa de calor
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", cbar=False)

    # Añadir títulos y etiquetas
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicción')
    plt.xticks(ticks=[0.5, 1.5], labels=['Positivo', 'Negativo'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Positivo', 'Negativo'])
    plt.show()

# crear secuencias de entrada para el modelo
def create_sequences(values, time_steps=10):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def plot_cut(df, signals_trend):

    # define colour palette and seaborn style
    pal = sns.cubehelix_palette(len(signals_trend), rot=-0.25, light=0.7)
    sns.set(style="white", context="notebook")

    fig, axes = plt.subplots(
        len(signals_trend), 1, dpi=150, figsize=(5, 8), sharex=True, constrained_layout=True,
    )


    # go through each of the signals
    for i in range(len(signals_trend)):
        # plot the signal
        # note, we take the length of the signal (9000 data point)
        # and divide it by the frequency (250 Hz) to get the x-axis
        # into seconds
        axes[i].plot(np.arange(0,len(df)),
                     df[signals_trend[i]],
                     color=pal[i],
                     linewidth=0.5,
                     alpha=1)

        axis_label = signals_trend[i]

        # Mostrar valores en el eje Y
        y_values = df[signals_trend[i]].values
        y_ticks = np.linspace(y_values.min(), y_values.max(), num=3)  # Ajusta según tus datos
        y_ticklabels = ["{:.2f}".format(value) if value <= 1 else "{:.0f}".format(value) for value in y_ticks]
        axes[i].set_yticks(y_ticks)
        axes[i].set_yticklabels(y_ticklabels, fontsize=5)

        axes[i].set_ylabel(
            axis_label, fontsize=7,
        )

        # if it's not the last signal on the plot
        # we don't want to show the subplot outlines
        if i != 5:
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["left"].set_visible(False)
            axes[i].spines["bottom"].set_visible(False)
            # axes[i].set_yticks([]) # also remove the y-ticks, cause ugly

        # for the last signal we will show the x-axis labels
        # which are the length (in seconds) of the signal
        else:
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["left"].set_visible(False)
            axes[i].spines["bottom"].set_visible(False)
            # axes[i].set_yticks([])
            axes[i].tick_params(axis="x", labelsize=7)
            axes[i].set_xlabel('Seconds', size=5)