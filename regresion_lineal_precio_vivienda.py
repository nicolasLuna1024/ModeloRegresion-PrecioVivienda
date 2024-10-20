#IMPORTAR LAS LIBRERIAS 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #Importar modelo de Regresion Lineal
from sklearn.metrics import mean_squared_error,r2_score #Importar Error Cuadratico Medio y Coeficiente de Determinacion 
import matplotlib.pyplot as plt #Grafica


#IMPORTAR LOS DATOS Y LEERLOS
datos = pd.read_excel("WShouse.xlsx", decimal=',')
print(datos.head())
print(datos.describe())

# SEPARAR CARACTERISTICAS(X) Y ETIQUETAS(Y)
x = datos[["Tamanio"]]
y = datos["Precio_venta"]

# DIVIDIR DATOS EN CONJUTO DE ENTRENAMINETO 70% Y DE PRUEBAS 30%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)

#CREAR EL MODELO DE REGRESION LINEAL
#Importaremos el algoritmo 
model = LinearRegression()

#ENTRENAR EL MODELO CON LOS DATOS DE TRAIN(x_train.y_train)
model.fit(x_train,y_train)

#REALIZAR LAS PREDICCIONES A PARTIR DE LOS DATOS DE TEST
y_result = model.predict(x_test)

#EVALUAR EL MODELO CON LAS METRICAS
#mse
erroCuadratico = mean_squared_error(y_test,y_result)
#r2 
r2= r2_score(y_test,y_result)
print(f"El error Cuadratico Medio es: {erroCuadratico} y el Error Coeficiente es {r2}" )

# ECUACION DE REGRESION ESTIMADA 
beta1 = model.coef_[0]
beta0 = model.intercept_
print(f"Ecuacion de regresion estimada: {beta1} + {beta0}")

# PREDICCION CON 2500 PIES CUADRADOS
nuevo_dato = pd.DataFrame([[2.50]], columns=["Tamanio"])
precioVenta = model.predict(nuevo_dato)
print(f'Estimacion de precio de venta de una Casa de 2500 pies: {precioVenta} ')
#GRAFICA 
plt.scatter(x,y,label= "Datos Reales",color= "green")
plt.plot(x_test,y_result,color = "red",label ="Modelo Lienal")
plt.title('Regresion Lineal')
plt.xlabel("Tamanio")
plt.ylabel("Precio de Venta")
plt.legend()
plt.show()

