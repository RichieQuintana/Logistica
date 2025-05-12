# K-Vecinos más Cercanos (K-NN)

#Explicación de los cambios:
#Entrenamiento y prueba: El script ahora está mejor documentado para que los estudiantes entiendan cómo dividir los datos en conjuntos de entrenamiento y prueba, y cómo realizar la predicción.
#Escalado de características: Se ha añadido un paso para normalizar los datos con StandardScaler, lo cual es importante para mejorar el rendimiento de los algoritmos basados en distancias, como el K-NN.
#Matriz de confusión: Se genera la matriz de confusión y se calcula la precisión del modelo para evaluar su rendimiento.
#Visualización: La visualización del conjunto de entrenamiento y prueba ahora incluye una visualización del modelo ajustado sobre la malla de características, lo que ayuda a entender cómo el modelo hace las predicciones.

# Regresión Logística con el dataset de desempeño estudiantil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargar el dataset
dataset = pd.read_csv('StudentsPerformance.csv')

# Mostrar las columnas disponibles
print("Columnas disponibles:")
print(dataset.columns)

# Crear la variable objetivo binaria: 1 si math score >= 60, 0 si no
dataset['pass_math'] = (dataset['math score'] >= 60).astype(int)

# Usar dos características numéricas: reading y writing scores
X = dataset[['reading score', 'writing score']].values
y = dataset['pass_math'].values

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalar características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenar modelo
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicción para un estudiante con reading=70, writing=75
resultado = classifier.predict(sc.transform([[70, 75]]))
print(f"\n¿Aprueba matemáticas con lectura=70 y escritura=75?: {resultado[0]}")

# Evaluar modelo
y_pred = classifier.predict(X_test)
print("\nPredicciones vs reales:")
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")

# Visualización - entrenamiento
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-5, stop=X_set[:, 0].max()+5, step=0.25),
                     np.arange(start=X_set[:, 1].min()-5, stop=X_set[:, 1].max()+5, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')))
plt.title('Regresión Logística (Entrenamiento)')
plt.xlabel('Reading Score')
plt.ylabel('Writing Score')
plt.legend(['No Aprobó', 'Aprobó'])
plt.show()

# Visualización - prueba
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-5, stop=X_set[:, 0].max()+5, step=0.25),
                     np.arange(start=X_set[:, 1].min()-5, stop=X_set[:, 1].max()+5, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green')))
plt.title('Regresión Logística (Prueba)')
plt.xlabel('Reading Score')
plt.ylabel('Writing Score')
plt.legend(['No Aprobó', 'Aprobó'])
plt.show()
