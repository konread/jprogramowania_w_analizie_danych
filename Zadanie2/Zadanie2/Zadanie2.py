import pandas as pd

dataset = pd.read_csv('AirQualityUCI.csv', ';')
dataset.drop("Date", axis=1, inplace=True)
dataset.drop("Time", axis=1, inplace=True)
dataset.drop("Unnamed: 15", axis=1, inplace=True)
dataset.drop("Unnamed: 16", axis=1, inplace=True)

percentAnalysis = [["ColumnName","Percent %"]]
for colName in dataset:
    missingValueCounter = 0
    for value in dataset[colName]:
        if value == -200:
            missingValueCounter += 1
    total=len(dataset[colName])
    percentAnalysis.append([colName,str(round((missingValueCounter/total)*100,2))])

for row in percentAnalysis:
    print("{: >15} {: >15}".format(*row))