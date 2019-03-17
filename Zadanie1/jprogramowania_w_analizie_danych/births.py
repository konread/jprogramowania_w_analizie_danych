import re as re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing

def main():
    data = pd.read_csv('births.csv', sep = ',')

    data['births'] = data['births'].astype('float64')

    hypothesis_value = 10000.0

    #normalized_hypothesis_value = preprocessing.normalize([[10000]])[0][0]
    #normalized_data = preprocessing.normalize([np.array(data['births'])])[0]
    #ttest = stats.ttest_1samp(a = normalized_data, popmean = normalized_hypothesis_value)

    plt.hist(data['births'], rwidth = 0.85, bins = 25, density = True)

    xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, len(data['births']))

    kde = stats.gaussian_kde(data['births'])

    plt.plot(x, kde(x), color = 'red')
    plt.xlabel('Number of births')
    plt.ylabel('Frequency')
    plt.annotate('Badana hipoteza', xy = (10000, 0.0001), xytext = (12000, 0.0001), arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.show()

    ttest = stats.ttest_1samp(a = data['births'], popmean = hypothesis_value)

    df = len(data['births']) - 1

    p = stats.t.cdf(x = ttest.statistic, df = df) * 2

    print('statistic: ' + str(ttest.statistic))
    print('pvalue: ' + str(ttest.pvalue))
    print('p: ' + str(p))

    alpha = 0.05

    if ttest.pvalue > alpha:
        print('Zaakceptuj hipotezę')
    else:
        print('Odrzuć hipotezę')

if __name__ == "__main__":
    main()