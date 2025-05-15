# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Configuración para visualizaciones - usando estilos compatibles con versiones actuales
# En lugar de 'seaborn-whitegrid' que ya no está disponible
sns.set_theme(style="whitegrid")  # Estilo actualizado
plt.rcParams['figure.figsize'] = (12, 8)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Cargar el dataset
    df = pd.read_csv('./dataset/Iris.csv')

    # 1. EXPLORACIÓN INICIAL
    print("=" * 50)
    print("EXPLORACIÓN INICIAL DEL DATASET")
    print("=" * 50)

    # Primeras filas del dataset
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())

    # Información del dataset
    print("\nInformación del dataset:")
    print(df.info())

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe().round(2))

    # Verificar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Distribución de especies
    print("\nDistribución de especies:")
    print(df['Species'].value_counts())

    # 2. ANÁLISIS UNIVARIADO
    print("\n" + "=" * 50)
    print("ANÁLISIS UNIVARIADO")
    print("=" * 50)

    # Eliminar la columna ID para análisis (si es solo un índice)
    df_analysis = df.drop('Id', axis=1)

    # Histogramas de las variables numéricas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, hue='Species', kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {col} por Especie')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig('./imagenes/histogramas_iris.png')
    plt.close()

    # Boxplots para detectar outliers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x='Species', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'Boxplot de {col} por Especie')
        axes[i].set_xlabel('Especie')
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.savefig('./imagenes/boxplots_iris.png')
    plt.close()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('EDA Iris')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
