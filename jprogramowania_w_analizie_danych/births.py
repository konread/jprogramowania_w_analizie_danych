import re as re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing

def main():
    data = pd.read_csv('births.csv', sep = ',')

    hypothesis_value = 10000.0

    #normalized_hypothesis_value = preprocessing.normalize([[10000]])[0][0]
    #normalized_data = preprocessing.normalize([np.array(data['births'])])[0]
    #ttest = stats.ttest_1samp(a = normalized_data, popmean = normalized_hypothesis_value)

    ttest = stats.ttest_1samp(a = data['births'], popmean = hypothesis_value)

    #p = stats.t.cdf(ttest.statistic, ttest.pvalue)

    print('statistic: ' + str(ttest.statistic))
    print('pvalue: ' + str(ttest.pvalue))
    #print('p: ' + str(p))

    alpha = 0.05

    if ttest.pvalue > alpha:
        print('Zaakceptuj hipotezę')
    else:
        print('Odrzuć hipotezę')

    plt.hist(data['births'], rwidth = 0.85)
    plt.xlabel('Number of births')
    plt.ylabel('Frequency')
    plt.annotate('Badana hipoteza', xy = (10000, 1200), xytext = (11500, 1000), arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.show()

if __name__ == "__main__":
    main()