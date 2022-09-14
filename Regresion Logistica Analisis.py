#Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
# Código para hacer análisis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

def sigmoide(Z):#Z entrada lineal
  """Función sigmoide"""
  return 1/(1+np.exp(-Z))

def iniciar_pesos_cero(longitudEntrada):#numero de atributos
  """Función para iniciar pesos y bias"""
  w = np.zeros((longitudEntrada, 1))
  b = 0
  return w, b

def forward_propagation(w, b, X): 
  """Función de forward propagation y = ax + b"""
  A = sigmoide(np.dot(w.T, X) + b) #predicción
  return A

def back_propagation(A, X, Y):
  """Función de backpropagation"""
  m = X.shape[1]
  wGrad = (1/m)*(np.dot(X, (A-Y).T)) #dervivada de w
  bGrad = (1/m)*(np.sum(A-Y))
  return wGrad, bGrad

def calcular_costo(A, Y): #predicción y salida #1 x numero de muestras
  """Función para calcular el costo ó error"""
  m = Y.shape[1]
  costo = -(1/m)*np.sum(Y*np.log(A)+((1-Y)*np.log(1-A)))
  return costo 

def descenso_gradiente(X, Y, w, b, learning_rate, epocas, 
                      mostrar_costo=False):
  """Función del descenso del gradiente"""
  costos = [] #Para graficar los costos y ver si converge
  for i in range(epocas):
    A = forward_propagation(w, b, X)
    costo = calcular_costo(A, Y)
    wGrad, bGrad =  back_propagation(A, X, Y)

    #Actualizar los pesos
    w = w - learning_rate * wGrad
    b = b - learning_rate * bGrad
    
    if i%100 == 0:
    #Almacenar los costos
      costos.append(costo)
    if mostrar_costo and i%100 == 0:
      print("Costo de la epoca " + str(i)+" : "+str(costo))
    
  #Guardar parámetros
  parametros = {"w":w, "b":b}
  return parametros, costos

def predecir(w, b, X):
  """Función para predecir con el modelo"""
  m = X.shape[1]
  resultadoY = np.zeros((1,m))
  w = w.reshape(X.shape[0],1)

  #Computamos propagazión hacia adelante
  A = forward_propagation(w, b, X) #predicciones
  for i in range(A.shape[1]): #m 
    #convertir valores flotantes a 0 o 1
    if (A[0][i] <= 0.5):
      resultadoY[0][i] = 0
    else:
      resultadoY[0][i] = 1
  return resultadoY

def regresion_logistica(entrenaX, entrenaY, pruebaX, pruebaY, epocas=2000,
                       learning_rate=0.005, depurar=False):
  """Función de regresión logística"""
  #Iniciar pesos
  w, b = iniciar_pesos_cero(entrenaX.shape[0])
  #Decenso del gradiente
  parametros, costos = descenso_gradiente(entrenaX, entrenaY, w, b,
                                            learning_rate, epocas, 
                                            mostrar_costo=depurar)
  #Devolver parámetros
  w = parametros["w"]
  b = parametros["b"]

  #Predecir los conjuntos de entrenamiento y prueba
  prediccionPrueba = predecir(w, b, pruebaX)
  prediccionEntrena = predecir(w, b, entrenaX)

  #Calcular la precisión
  precisionEntrena = 100-np.mean((np.abs(prediccionEntrena-entrenaY))*100)
  precisionPruebas = 100-np.mean((np.abs(prediccionPrueba-pruebaY))*100)

  #Guardar los resultados
  resultados = {"costos":costos,
                "w":w, "b":b, "learningRate":learning_rate,
                "epocas": epocas, "precisionPruebas":precisionPruebas,
                "precisionEntrenamiento": precisionEntrena}
  return resultados


def main():
    #Cargar el dataset
    df = pd.read_csv("winequality_red.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    #Escalamiento
    for i in X:
      X[i] = (X[i] - X[i].min()) / (X[i].max() - X[i].min())

    #Separar variables en train, test y split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
      test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
      test_size=0.25) # 0.25 x 0.8 = 0.2

    #Transformar  variables a un array de numpy
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()


    #Generar learning_rates aleatorios
    learning_rates = list(np.random.rand(4,))
    #Generar learning_rates epocas
    epocas = list(np.random.rand(4,) * 10000)
    epocas = [int(i) for i in epocas]

    #Preparar las variables x y y para los calculos matriciales
    X_train = X_train.T
    y_train = y_train.reshape(1, X_train.shape[1])
    X_val = X_val.T
    y_val = y_val.reshape(1, X_val.shape[1])

    #Configuraciones para graficar
    _, axis = plt.subplots(2, 2)
    #Lista para guardar la precision de entrenamiento más alta
    maxx = [-1000, 0]
    resultado = 0 #variable para poder accederla los resultados afuera del for
    j = 0 #Contador para graficar en diferentes subplots
    for alpha, epoca in zip(learning_rates, epocas):
      #Entrenar modelo
      resultado = regresion_logistica(X_train, y_train, X_val, y_val,
                                      epocas=epoca, depurar=True,
                                      learning_rate=alpha)      
      if resultado["precisionEntrenamiento"] > maxx[0]:
        maxx = [resultado["precisionEntrenamiento"], alpha, epoca]
      
      #Condición para graficar 4 veces
      y_pred = []
      cont = 0
      for i in X_test:
        prueba_1 = i.reshape(11, 1)
        prediccionPrueba = predecir(resultado["w"], resultado["b"], prueba_1)
        if j == 0:
          i1, i2 = 0,0
        elif j == 1:
          i1, i2 = 0,1
        elif j == 2:
          i1, i2 = 1,0
        else:
          i1, i2 = 1,1

        if prediccionPrueba == y_test[cont]:
          axis[i1, i2].scatter(cont, prediccionPrueba, color='green')
        else:
          axis[i1, i2].scatter(cont,prediccionPrueba, color='red')
          axis[i1, i2].scatter(cont,prediccionPrueba, color='red')
        cont += 1
      j += 1
    
    plt.ylabel("Predicción")
    plt.xlabel("Muestra")
    plt.show()

    #Graficar los costos
    costos = np.squeeze(resultado["costos"]) 
    plt.plot(costos)
    plt.ylabel("Costo")
    plt.xlabel("Prueba")
    plt.title(f"Learning Rate = {resultado['learningRate']}")
    plt.show()

    #Generar nuevos rangos entre los hiper parametros
    #Rango de learning rates
    maxLeftLR = maxx[1] - .05
    maxRightLR = maxx[1] + .05
    rangeLR1 = np.linspace(maxLeftLR, maxx[1], num=10)
    rangeLR2 = np.linspace(maxx[1],maxRightLR, num=10)
    rangeLR3 = [y for x in [rangeLR1, rangeLR2] for y in x]
    rangeLR3 = list(dict.fromkeys(rangeLR3))#Eliminar repetidos

    #Rango de épocas
    maxLeftEpoc = maxx[2] - 20
    maxRightEpoc = maxx[2] + 20
    rangeEpoc1 = np.linspace(maxLeftEpoc, maxx[2], num=20)
    rangeEpoc1 = [int(i) for i in rangeEpoc1]
    rangeEpoc2 = np.linspace(maxx[2], maxRightEpoc, num=10)
    rangeEpoc2 = [int(i) for i in rangeEpoc2]
    rangeEpoc3 = [y for x in [rangeEpoc1, rangeEpoc2] for y in x]
    rangeEpoc3 = list(dict.fromkeys(rangeEpoc3)) #Eliminar repetidos

    #Nueva iteración Coarse to fine
    maxx1 = [-1000, 0]
    for alpha, epoca in zip(rangeLR3, rangeEpoc3):
        resultado = regresion_logistica(X_train, y_train, X_val, y_val,
                                        epocas=epoca, depurar=True,
                                        learning_rate=alpha)
        if resultado["precisionEntrenamiento"] > maxx1[0]:
            maxx1 = [resultado["precisionEntrenamiento"], alpha, epoca]

    #Escoger los hiper parametros (1ra iteración y con coarse to fine)
    #que tuvieron un mejor precision
    best_hyperparameters = maxx if maxx[0] > maxx1[0] else maxx1
    print(best_hyperparameters)
    _, alpha, epoca = best_hyperparameters

    
    w = resultado["w"]
    b = resultado["b"]

    #Predecir con conjunto test y graficar
    _, axis = plt.subplots(2, 2)
    for j in range(4):
      y_pred = []
      cont = 0
      for i in X_test:
        prueba_1 = i.reshape(11, 1)
        prediccionPrueba = predecir(w, b, prueba_1)
        y_pred.append(np.squeeze(np.squeeze(prediccionPrueba)))

        if j == 0:
          i1,i2 = 0,0
        elif j ==1:
          i1,i2 = 0,1
        elif j ==2:
          i1,i2 = 1,0
        else:
          i1,i2 = 1,1

        #Scatter
        if prediccionPrueba == y_test[cont]:
          axis[i1, i2].scatter(cont, prediccionPrueba, color='green')
        else:
          axis[i1, i2].scatter(cont,prediccionPrueba, color='red')
        cont += 1
    plt.show()
    
    # Scores
    acs, pc = accuracy_score(y_pred, y_test), precision_score(y_pred, y_test)
    print("Scores")
    print("Accuracy Score", acs)
    print("Precision Score", pc)

    # Costos
    costos = np.squeeze(resultado["costos"])
    plt.plot(costos)
    plt.ylabel("Costo")
    plt.xlabel("Epocas (cada 100)")
    plt.title(f"Learning Rate = {resultado['learningRate']}")
    plt.show()

if __name__ == "__main__":
    main()