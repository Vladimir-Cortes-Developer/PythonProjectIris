# Análisis Exploratorio de Datos: Dataset Iris

# Bootcamp en Inteligencia artificial (Talento Tech)
## Nivel: Explorador - Básico-2025-5-L2-G47
## Realizado por:  Víctor C. Vladimir Cortés A.

## Descripción del Proyecto

Este proyecto realiza un análisis exploratorio de datos (EDA) completo sobre el famoso dataset Iris. El conjunto de datos contiene mediciones de cuatro características morfológicas de flores (longitud y ancho de sépalos y pétalos) para tres especies diferentes de Iris: setosa, versicolor y virginica.

El análisis incluye estadísticas descriptivas, visualizaciones, análisis univariado, bivariado y multivariado, así como pruebas estadísticas para identificar patrones y relaciones entre las variables.

## Requisitos

El proyecto requiere Python 3.13.3+ y las siguientes bibliotecas:

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Componentes del Análisis

El análisis exploratorio de datos incluye:

### 1. Exploración Inicial
* Revisión de las primeras filas del dataset
* Información sobre la estructura de los datos
* Estadísticas descriptivas
* Verificación de valores nulos
* Distribución de especies

### 2. Análisis Univariado
* Histogramas de cada característica por especie
* Boxplots para identificar outliers

### 3. Análisis Bivariado
* Matriz de correlación entre características
* Visualización de correlaciones mediante heatmap
* Scatter plots entre pares de variables

### 4. Análisis Multivariado
* Visualización 3D de las características más discriminativas

### 5. Análisis Estadístico
* ANOVA para comparar medias entre especies

### 6. Conclusiones
* Resumen de los principales hallazgos
* Patrones identificados
* Implicaciones para tareas de clasificación

## Resultados Principales

El análisis revela que:

1. Las tres especies están equilibradas en el dataset (50 muestras de cada una).
2. No existen valores nulos en los datos.
3. Las características relacionadas con los pétalos (longitud y ancho) son más discriminativas que las de los sépalos.
4. Iris setosa es claramente separable de las otras dos especies.
5. Existe cierto solapamiento entre Iris versicolor e Iris virginica.
6. Hay una fuerte correlación positiva entre la longitud y el ancho del pétalo.
7. Todas las características presentan diferencias estadísticamente significativas entre especies (comprobado mediante ANOVA).

## Visualizaciones Generadas

* **histogramas_iris.png**: Distribución de cada característica por especie
* **boxplots_iris.png**: Identificación de outliers y comparación de la distribución por especie
* **correlacion_iris.png**: Heatmap de correlación entre variables numéricas
* **pairplot_iris.png**: Matriz de dispersión entre todas las variables
* **3d_scatter_iris.png**: Visualización tridimensional de la separación entre especies


Puedes instalar todas las dependencias con:


```bash
pip install -r requirements.txt