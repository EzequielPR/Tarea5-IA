import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error

# link al dataset
# http://onapi.gob.do/wp-content/uploads/datos/Estadisticas%20Invenciones/IN-2000-2021.xlsx
# convertido a .csv con Excel

# importo los datos del archivo "estadisticas de invenciones, 2000-2021.csv"
dataset = pd.read_csv("estadisticas de invenciones, 2000-2021.csv", delimiter=";")
# los convierto a un dataframe
datos = pd.DataFrame(dataset)

# filtro los datos por el tipo "Patente de Invención", ya que el archivo .csv trae 3 tipos de solicitud
datos = datos[datos["Tipo de Solicitud"] == "Patente de Invención"]

# guardo todos los años en x, y el acumulado de patentes en y
x = datos["Año"]
y = datos["Cantidad"].cumsum()


# creo esta variable X que es un arreglo de arreglos ya que asi necesita LinearRegression el eje x
X = x[:,np.newaxis]
# el modelo LinearRegression crea la regresion lineal
reg = linear_model.LinearRegression().fit(X, y)

# formula de la recta => mx+b
## crea el coeficiente m o de inclinacion mediante los valores de x e y
m = reg.coef_[0]
## crea el coeficiente b o de intercepcion mediante los valores de x e y
b = reg.intercept_

# predice los valores de y utilizando la formula de la recta
y_p = m*X+b


# imprimo la recta de tendencia, el coeficiente de determinacion, y el error absoluto
print("\nLinea de Tendencia\n")
formula = f"y = {m}*x + {b}" if b >= 0 else f"y = {m}*x - {abs(b)}"
r2 = f"R² = {r2_score(y, y_p)}"
error_abs = f"error absoluto = {mean_absolute_error(y, y_p)}"
print(f"{formula}\n{r2}\n{error_abs}\n")

# imprimo los valores actuales y los de la recta asi como la diferencia entre estos
print("\nTabla de Valores\n")
print("año \t| actual | pronosticado | diferencia")
for año in x:
    i = año%2000
    valor_real = y[i]
    valor_predicho = int( m*año+b )
    diferencia = abs(valor_real - abs(valor_predicho))
    print(f"{año} \t| {valor_real} \t| {valor_predicho} \t| {diferencia}")
print("")

# imprimo los valores predichos para los siguientes 5 años
print("\nPronosticos para los siguientes 5 años\n")
siguiente_año = x[len(x)-1]+1
mas_años = [año for año in range(siguiente_año, siguiente_año+5)]
print("año \t| pronosticado")
for año in mas_años:
    valor_predicho = int( m*año+b )
    print( f"{año} \t| {valor_predicho}" )
print()


# agrego los datos a la grafica
plt.plot(x, y, ".", label="patentes acumuladas")
plt.plot(x, y_p, label="linea de tendencia")
plt.legend()

# agrego el titulo y las etiquetas a los ejes
plt.title("Solicitudes de Patentes de Invenciones 2000 - 2021")
plt.xlabel("año")
plt.ylabel("patentes")

# muestra el grafico
plt.show()
