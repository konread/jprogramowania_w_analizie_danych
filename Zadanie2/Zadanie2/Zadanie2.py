import re as re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('AirQualityUCI.csv', ';')
dataset.drop("Date", axis=1, inplace=True)
dataset.drop("Time", axis=1, inplace=True)
dataset.drop("NMHC(GT)", axis=1, inplace=True)
dataset.drop("Unnamed: 15", axis=1, inplace=True)
dataset.drop("Unnamed: 16", axis=1, inplace=True)

header = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

dataset[header] = dataset[header].replace(-200, np.NaN)

percent_nan = dataset.isnull().sum() * 100 / len(dataset)
   
print("Procent wartości pustych dla [PT08.S1(CO)]: " + str(percent_nan['PT08.S1(CO)']))
print("Procent wartości pustych dla [C6H6(GT)]: " + str(percent_nan['C6H6(GT)']))

dataset_imputer = dataset.copy()
dataset_not_nan = dataset.copy()

imp_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')

values = dataset_imputer.values
transformed_values = imp_mean.fit_transform(values)

dataset_imputer[header] = transformed_values

x_imputer = dataset_imputer['PT08.S1(CO)']
y_imputer = dataset_imputer['C6H6(GT)']

slope, intercept, r_value, p_value, std_err = stats.linregress(x_imputer, y_imputer)

slope_imputer = slope
intercept_imputer = intercept
r_value_imputer = r_value
p_value_imputer = p_value
std_err_imputer = std_err

plt.plot(x_imputer, y_imputer, 'o', label = 'dane (po imputacji)')
plt.plot(x_imputer, intercept + slope * x_imputer, 'r', label = 'regresja liniowa (po imputacji)', color='green')
plt.legend()
#plt.show()

x_not_nan = dataset_not_nan['PT08.S1(CO)'].dropna()
y_not_nan = dataset_not_nan['C6H6(GT)'].dropna()

slope, intercept, r_value, p_value, std_err = stats.linregress(x_not_nan, y_not_nan)

slope_not_nan = slope
intercept_not_nan = intercept
r_value_not_nan = r_value
p_value_not_nan = p_value
std_err_not_nan = std_err

plt.plot(x_not_nan, y_not_nan, 'o', label='dane (z brakami)')
plt.plot(x_not_nan, intercept + slope * x_not_nan, 'r', label='regresja liniowa (z brakami)')
plt.legend()
#plt.show()

# -- Zadanie na 4 --

#Metoda interpolacji

dataset_interpolate = dataset.copy()

dataset_interpolate[header] = dataset_interpolate.interpolate(method="linear")

print(dataset_interpolate)

x_interpolate = dataset_interpolate['PT08.S1(CO)']
y_interpolate = dataset_interpolate['C6H6(GT)']

slope, intercept, r_value, p_value, std_err = stats.linregress(x_interpolate, y_interpolate)

slope_interpolate = slope
intercept_interpolate = intercept
r_value_interpolate = r_value
p_value_interpolate = p_value
std_err_interpolate = std_err

plt.plot(x_interpolate, y_interpolate, 'o', label = 'dane (po interpolacji)')
plt.plot(x_interpolate, intercept + slope * x_interpolate, 'r', label = 'regresja liniowa (po interpolacji)', color='blue')
plt.legend()
#plt.show()

print("---------------------------------------------")
print("Nachylenie linii regresji (nan): " + str(slope_not_nan))
print("Nachylenie linii regresji (mean imputation): " + str(slope_imputer))
print("Nachylenie linii regresji (interpolate): " + str(slope_interpolate))
print()
print("Współczynnik korelacji (nan): " + str(r_value_not_nan))
print("Współczynnik korelacji (mean imputation): " + str(r_value_imputer))
print("Współczynnik korelacji (interpolate): " + str(r_value_interpolate))
print()
print("Błąd standrardowy (nan): " + str(std_err_not_nan))
print("Błąd standrardowy (mean imputation): " + str(std_err_imputer))
print("Błąd standrardowy (interpolate): " + str(std_err_interpolate))
print("---------------------------------------------")
print("Średnia x (nan): " + str(dataset['PT08.S1(CO)'].mean()))
print("Średnia x (mean imputation): " + str(x_imputer.mean()))
print("Średnia x (interpolate): " + str(x_interpolate.mean()))
print()
print("Średnia y (nan): " + str(dataset['C6H6(GT)'].mean()))
print("Średnia y (mean imputation): " + str(y_imputer.mean()))
print("Średnia y (interpolate): " + str(y_interpolate.mean()))
print("---------------------------------------------")
print("Odchylenie standardowe x (nan): " + str(dataset['PT08.S1(CO)'].std()))
print("Odchylenie standardowe x (mean imputation): " + str(x_imputer.std()))
print("Odchylenie standardowe x (interpolate): " + str(x_interpolate.std()))
print()
print("Odchylenie standardowe y (nan): " + str(dataset['C6H6(GT)'].std()))
print("Odchylenie standardowe y (mean imputation): " + str(y_imputer.std()))
print("Odchylenie standardowe y (interpolate): " + str(y_interpolate.std()))
print("---------------------------------------------")
print("Kwartyle x (nan): " + str(np.percentile(x_not_nan, [25, 25, 25, 25])))
print("Kwartyle x (mean imputation): " + str(np.percentile(x_imputer, [25, 25, 25, 25])))
print("Kwartyle x (interpolate): " + str(np.percentile(x_interpolate, [25, 25, 25, 25])))
print()
print("Kwartyle y (nan): " + str(np.percentile(y_not_nan, [25, 25, 25, 25])))
print("Kwartyle y (mean imputation): " + str(np.percentile(y_imputer, [25, 25, 25, 25])))
print("Kwartyle y (interpolate): " + str(np.percentile(y_interpolate, [25, 25, 25, 25])))
print("---------------------------------------------")