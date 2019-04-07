import re as re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import math
from operator import itemgetter

def hotDeckImputer(dataset, columnName):
    dataset_with_missig_values = dataset[np.isnan(dataset[columnName])]
    dataset_without_missig_values = dataset[dataset[columnName].notnull()]
    rows, columns = dataset_with_missig_values.shape
    datasetColumn = dataset[columnName].values
    computedMissingValues = []

    # For each missing element in data set
    for idxMissingValues, rowMissingValues in enumerate(dataset_with_missig_values.values):
        euclidean = []
        # Loop all non-missing rows...
        for idxNoneMissValues, rowNoneMissValues in enumerate(dataset_without_missig_values.values):
            euclideanTotal = 0
            # ...and all of its columns...
            for idxCol in range(columns):
                # ...except itself...
                rowMissingValue = rowMissingValues[idxCol]
                rowNoneMissValue = rowNoneMissValues[idxCol]
                if not math.isnan(rowMissingValue) and not math.isnan(rowNoneMissValue):
                    # ...to calculate the euclidean distance of both...
                    euclideanTotal += (rowMissingValue - rowNoneMissValue) ** 2

            dist = math.sqrt(euclideanTotal)
            # Append found euclidean and index of that in the original data set
            if dist != 0.0:
                euclidean.append((dist, idxNoneMissValues))

        # Sorts the euclidean list by their first value
        euclidean = sorted(euclidean, key=itemgetter(0))

        if not euclidean:
            computedMissingValues.append(0.0)

        for euclideanValue, euclideanIdx in euclidean:
            columnValues = dataset_without_missig_values[columnName].values
            value = columnValues[euclideanIdx]
            if not math.isnan(value):
                computedMissingValues.append(value)
                break

    idx = 0
    for idxDsCol, value in enumerate(datasetColumn):
        if math.isnan(value):
            datasetColumn[idxDsCol] = computedMissingValues[idx]
            idx += 1

    return datasetColumn


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
#plt.show()

# Metoda hot-deck
x_hot_deck = hotDeckImputer(dataset.copy(), 'PT08.S1(CO)')
y_hot_deck = hotDeckImputer(dataset.copy(), 'C6H6(GT)')

slope_hot_deck, intercept_hot_deck, r_value_hot_deck, p_value_hot_deck, std_err_hot_deck = stats.linregress(x_hot_deck, y_hot_deck)

plt.plot(x_hot_deck, y_hot_deck, 'o', label = 'dane (po hot-deck)')
plt.plot(x_hot_deck, intercept_hot_deck + slope_hot_deck * x_hot_deck, 'r', label = 'regresja liniowa (po hot-deck)', color='orange')
plt.legend()
#plt.show()

# Wartosci uzyskane z krzywej regresji z punktu 6
dataset_regression_values = dataset.copy()
x_regression = dataset_regression_values['PT08.S1(CO)'].values
y_regression = dataset_regression_values['C6H6(GT)'].values

for idx, value in enumerate(x_regression):
    x_value = x_regression[idx]
    y_value = y_regression[idx]
    if math.isnan(x_value) and math.isnan(y_value):
        x_regression[idx] = 0.0
        y_regression[idx] = 0.0

    if math.isnan(x_value) and not math.isnan(y_value):
        value = intercept_imputer_mean + (slope_imputer_mean * y_value) + std_err_imputer_mean
        x_regression[idx] = value

    if not math.isnan(x_value) and math.isnan(y_value):
        value = intercept_imputer_mean + (slope_imputer_mean * x_value) + std_err_imputer_mean
        y_regression[idx] = value

slope_regression, intercept_regression, r_value_regression, p_value_regression, std_err_regression = stats.linregress(x_regression, y_regression)

plt.plot(x_regression, y_regression, 'o', label = 'dane (po krzywej regresji)')
plt.plot(x_regression, intercept_regression + slope_regression * x_regression, 'r', label = 'regresja liniowa (po krzywej regresji)', color='gray')
plt.legend()
plt.show()

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

result["Metoda"].append("hot-deck")
result["Nachylenie linii regresji"].append(slope_hot_deck)
result["Wspolczynnik korelacji"].append(r_value_hot_deck)
result["Blad standrardowy"].append(std_err_hot_deck)
result["Srednia x"].append(x_hot_deck.mean())
result["Srednia y"].append(y_hot_deck.mean())
result["Odchylenie standardowe x"].append(x_hot_deck.std())
result["Odchylenie standardowe y"].append(y_hot_deck.std())
result["Kwartyle x"].append(np.percentile(x_hot_deck, [25, 25, 25, 25]))
result["Kwartyle y"].append(np.percentile(y_hot_deck, [25, 25, 25, 25]))

result["Metoda"].append("Z krzywej regresji")
result["Nachylenie linii regresji"].append(slope_regression)
result["Wspolczynnik korelacji"].append(r_value_regression)
result["Blad standrardowy"].append(std_err_regression)
result["Srednia x"].append(x_regression.mean())
result["Srednia y"].append(y_regression.mean())
result["Odchylenie standardowe x"].append(x_regression.std())
result["Odchylenie standardowe y"].append(y_regression.std())
result["Kwartyle x"].append(np.percentile(x_regression, [25, 25, 25, 25]))
result["Kwartyle y"].append(np.percentile(y_regression, [25, 25, 25, 25]))

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