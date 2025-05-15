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

    # 3. ANÁLISIS BIVARIADO
    print("\n" + "=" * 50)
    print("ANÁLISIS BIVARIADO")
    print("=" * 50)

    # Matriz de correlación
    corr_matrix = df_analysis.drop('Species', axis=1).corr().round(2)
    print("\nMatriz de correlación:")
    print(corr_matrix)

    # Visualizar matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlación de Variables')
    plt.tight_layout()
    plt.savefig('./imagenes/correlacion_iris.png')
    plt.close()

    # Scatter plots
    sns.pairplot(df, hue='Species', markers=['o', 's', 'D'], height=2.5)
    plt.suptitle('Scatter Plot Matrix para Variables del Dataset Iris', y=1.02)
    plt.savefig('./imagenes/pairplot_iris.png')
    plt.close()

    # 4. ANÁLISIS MULTIVARIADO
    print("\n" + "=" * 50)
    print("ANÁLISIS MULTIVARIADO")
    print("=" * 50)

    # Visualización 3D
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Colores para cada especie
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
    markers = {'Iris-setosa': 'o', 'Iris-versicolor': '^', 'Iris-virginica': 's'}

    for species, group in df.groupby('Species'):
        ax.scatter(group['SepalLengthCm'], group['PetalLengthCm'], group['PetalWidthCm'],
                   c=colors[species], marker=markers[species], s=60, label=species, alpha=0.7)

    ax.set_xlabel('Longitud de Sépalo (cm)')
    ax.set_ylabel('Longitud de Pétalo (cm)')
    ax.set_zlabel('Ancho de Pétalo (cm)')
    ax.set_title('Visualización 3D de Características del Iris')
    ax.legend()
    plt.savefig('./imagenes/3d_scatter_iris.png')
    plt.close()

    # 5. ANÁLISIS ESTADÍSTICO
    print("\n" + "=" * 50)
    print("ANÁLISIS ESTADÍSTICO")
    print("=" * 50)

    # ANOVA para comparar medias entre especies
    print("\nResultados de ANOVA para comparar características entre especies:")
    for feature in numeric_cols:
        groups = [df[df['Species'] == species][feature] for species in df['Species'].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"{feature}: F={f_stat:.4f}, p-valor={p_value:.8f}")

        # 6. CONCLUSIONES
        print("\n" + "=" * 50)
        print("CONCLUSIONES DEL EDA")
        print("=" * 50)

        print("""
        Principales hallazgos del análisis exploratorio:

        1. El dataset contiene 150 muestras de flores Iris, equitativamente distribuidas entre 3 especies.
        2. No hay valores nulos en el dataset.
        3. Las características PetalLengthCm y PetalWidthCm muestran una mayor capacidad discriminativa entre especies.
        4. Existe una fuerte correlación positiva entre la longitud y el ancho del pétalo.
        5. Iris-setosa es claramente separable de las otras dos especies.
        6. Iris-versicolor e Iris-virginica muestran cierto solapamiento en sus características.
        7. El análisis ANOVA confirma que todas las características presentan diferencias estadísticamente significativas entre especies.
        8. El dataset es adecuado para tareas de clasificación debido a la clara separación entre especies.
        """)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('EDA Iris')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
