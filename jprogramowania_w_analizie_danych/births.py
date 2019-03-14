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

    ttest = stats.ttest_1samp(a = data['births'], popmean = hypothesis_value)

    print(stats.t.cdf(ttest.statistic, ttest.pvalue))

    plt.hist(data['births'], rwidth = 0.85)
    plt.xlabel('Number of births')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()