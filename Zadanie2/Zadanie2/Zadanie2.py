import re as re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# -- Zadanie na 3 --
dataset = pd.read_csv('AirQualityUCI.csv', ';')
dataset.drop("Date", axis=1, inplace=True)
dataset.drop("Time", axis=1, inplace=True)
dataset.drop("NMHC(GT)", axis=1, inplace=True)
dataset.drop("Unnamed: 15", axis=1, inplace=True)
dataset.drop("Unnamed: 16", axis=1, inplace=True)

header = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

#Policzyc jaki procent danych zawiera braki.
dataset[header] = dataset[header].replace(-200, np.NaN)
percent_nan = dataset.isnull().sum() * 100 / len(dataset)
print("Procent wartości pustych dla [PT08.S1(CO)]: " + str(percent_nan['PT08.S1(CO)']))
print("Procent wartości pustych dla [C6H6(GT)]: " + str(percent_nan['C6H6(GT)']))

#Wyznaczyc krzywa regresji dla danych bez brakow.
dataset_not_nan = dataset.copy()
x_not_nan = dataset_not_nan['PT08.S1(CO)'].dropna()
y_not_nan = dataset_not_nan['C6H6(GT)'].dropna()

slope_not_nan, intercept_not_nan, r_value_not_nan, p_value_not_nan, std_err_not_nan = stats.linregress(x_not_nan, y_not_nan)

plt.plot(x_not_nan, y_not_nan, 'o', label='dane (bez braków)')
plt.plot(x_not_nan, intercept_not_nan + slope_not_nan * x_not_nan, 'r', label='regresja liniowa (bez braków)')
plt.legend()
#plt.show()

#Uzupelnic braki metoda "mean imputation".
dataset_imputer = dataset.copy()
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')

values = dataset_imputer.values
transformed_values = imp_mean.fit_transform(values)

dataset_imputer[header] = transformed_values

x_imputer_mean = dataset_imputer['PT08.S1(CO)']
y_imputer_mean = dataset_imputer['C6H6(GT)']

#krzywa regresji dla danych po imputacji
slope_imputer_mean, intercept_imputer_mean, r_value_imputer_mean, p_value_imputer_mean, std_err_imputer_mean = stats.linregress(x_imputer_mean, y_imputer_mean)

plt.plot(x_imputer_mean, y_imputer_mean, 'o', label = 'dane (po imputacji)')
plt.plot(x_imputer_mean, intercept_imputer_mean + slope_imputer_mean * x_imputer_mean, 'r', label = 'regresja liniowa (po imputacji)', color='green')
plt.legend()
#plt.show()

# -- Zadanie na 4 --

#Metoda interpolacji

dataset_interpolate = dataset.copy()
dataset_interpolate[header] = dataset_interpolate.interpolate(method="linear")

x_interpolate = dataset_interpolate['PT08.S1(CO)']
y_interpolate = dataset_interpolate['C6H6(GT)']

slope_interpolate, intercept_interpolate, r_value_interpolate, p_value_interpolate, std_err_interpolate = stats.linregress(x_interpolate, y_interpolate)

plt.plot(x_interpolate, y_interpolate, 'o', label = 'dane (po interpolacji)')
plt.plot(x_interpolate, intercept_interpolate + slope_interpolate * x_interpolate, 'r', label = 'regresja liniowa (po interpolacji)', color='blue')
plt.legend()
plt.show()

#print("---------------------------------------------")
#print("Nachylenie linii regresji (without nan): " + str(slope_not_nan))
#print("Nachylenie linii regresji (mean imputation): " + str(slope_imputer_mean))
#print("Nachylenie linii regresji (interpolate): " + str(slope_interpolate))
#print()
#print("Współczynnik korelacji (without nan): " + str(r_value_not_nan))
#print("Współczynnik korelacji (mean imputation): " + str(r_value_imputer_mean))
#print("Współczynnik korelacji (interpolate): " + str(r_value_interpolate))
#print()
#print("Błąd standrardowy (without nan): " + str(std_err_not_nan))
#print("Błąd standrardowy (mean imputation): " + str(std_err_imputer_mean))
#print("Błąd standrardowy (interpolate): " + str(std_err_interpolate))
#print("---------------------------------------------")
#print("Średnia x (without nan): " + str(dataset['PT08.S1(CO)'].mean()))
#print("Średnia x (mean imputation): " + str(x_imputer_mean.mean()))
#print("Średnia x (interpolate): " + str(x_interpolate.mean()))
#print()
#print("Średnia y (without nan): " + str(dataset['C6H6(GT)'].mean()))
#print("Średnia y (mean imputation): " + str(y_imputer_mean.mean()))
#print("Średnia y (interpolate): " + str(y_interpolate.mean()))
#print("---------------------------------------------")
#print("Odchylenie standardowe x (without nan): " + str(dataset['PT08.S1(CO)'].std()))
#print("Odchylenie standardowe x (mean imputation): " + str(x_imputer_mean.std()))
#print("Odchylenie standardowe x (interpolate): " + str(x_interpolate.std()))
#print()
#print("Odchylenie standardowe y (without nan): " + str(dataset['C6H6(GT)'].std()))
#print("Odchylenie standardowe y (mean imputation): " + str(y_imputer_mean.std()))
#print("Odchylenie standardowe y (interpolate): " + str(y_interpolate.std()))
#print("---------------------------------------------")
#print("Kwartyle x (without nan): " + str(np.percentile(x_not_nan, [25, 25, 25, 25])))
#print("Kwartyle x (mean imputation): " + str(np.percentile(x_imputer_mean, [25, 25, 25, 25])))
#print("Kwartyle x (interpolate): " + str(np.percentile(x_interpolate, [25, 25, 25, 25])))
#print()
#print("Kwartyle y (without nan): " + str(np.percentile(y_not_nan, [25, 25, 25, 25])))
#print("Kwartyle y (mean imputation): " + str(np.percentile(y_imputer_mean, [25, 25, 25, 25])))
#print("Kwartyle y (interpolate): " + str(np.percentile(y_interpolate, [25, 25, 25, 25])))
#print("---------------------------------------------")

result = dict()

result["Metoda"] = []
result["Nachylenie linii regresji"] = []
result["Wspolczynnik korelacji"] = []
result["Blad standrardowy"] = []
result["Srednia x"] = []
result["Srednia y"] = []
result["Odchylenie standardowe x"] = []
result["Odchylenie standardowe y"] = []
result["Kwartyle x"] = []
result["Kwartyle y"] = []

result["Metoda"].append("without nan")
result["Nachylenie linii regresji"].append(slope_not_nan)
result["Wspolczynnik korelacji"].append(r_value_not_nan)
result["Blad standrardowy"].append(std_err_not_nan)
result["Srednia x"].append(dataset['PT08.S1(CO)'].mean())
result["Srednia y"].append(dataset['C6H6(GT)'].mean())
result["Odchylenie standardowe x"].append(dataset['PT08.S1(CO)'].std())
result["Odchylenie standardowe y"].append(dataset['C6H6(GT)'].std())
result["Kwartyle x"].append(np.percentile(x_not_nan, [25, 25, 25, 25]))
result["Kwartyle y"].append(np.percentile(y_not_nan, [25, 25, 25, 25]))

result["Metoda"].append("mean imputation")
result["Nachylenie linii regresji"].append(slope_imputer_mean)
result["Wspolczynnik korelacji"].append(r_value_imputer_mean)
result["Blad standrardowy"].append(std_err_imputer_mean)
result["Srednia x"].append(x_imputer_mean.mean())
result["Srednia y"].append(y_imputer_mean.mean())
result["Odchylenie standardowe x"].append(x_imputer_mean.std())
result["Odchylenie standardowe y"].append(y_imputer_mean.std())
result["Kwartyle x"].append(np.percentile(x_imputer_mean, [25, 25, 25, 25]))
result["Kwartyle y"].append(np.percentile(y_imputer_mean, [25, 25, 25, 25]))

result["Metoda"].append("interpolate")
result["Nachylenie linii regresji"].append(slope_interpolate)
result["Wspolczynnik korelacji"].append(r_value_interpolate)
result["Blad standrardowy"].append(std_err_interpolate)
result["Srednia x"].append(x_interpolate.mean())
result["Srednia y"].append(y_interpolate.mean())
result["Odchylenie standardowe x"].append(x_interpolate.std())
result["Odchylenie standardowe y"].append(y_interpolate.std())
result["Kwartyle x"].append(np.percentile(x_interpolate, [25, 25, 25, 25]))
result["Kwartyle y"].append(np.percentile(y_interpolate, [25, 25, 25, 25]))


df = pd.DataFrame(result ,columns= ['Metoda', 
                                    'Nachylenie linii regresji', 
                                    'Wspolczynnik korelacji', 
                                    'Blad standrardowy', 
                                    'Srednia x', 
                                    'Srednia y', 
                                    'Odchylenie standardowe x', 
                                    'Odchylenie standardowe y', 
                                    'Kwartyle x', 
                                    'Kwartyle y'])

df.to_csv('result.csv')