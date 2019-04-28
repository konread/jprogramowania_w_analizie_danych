import re as re
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from time import time
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle
from random import Random
from warnings import filterwarnings

filterwarnings('ignore')

def classifier(train_x, test_x, train_y, test_y):
    step = 1
    max = 100
    iter = 0
    result = []
    
    while iter < max:
        model = svm.LinearSVC(max_iter = float(iter))
        model.fit(train_x, train_y.values.ravel())
        predictions_test = model.predict(test_x)
        #model_score = model.score(test_x, test_y.values.ravel())
        model_score = accuracy_score(test_y, predictions_test)

        result.append(model_score)
    
        iter += step
    
    return result

def main():
    names = ["Cultivar", "Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline"]

    dataset = pd.read_csv('data.csv', names = names)

    df = pd.DataFrame(dataset)

    scaler = MinMaxScaler()

    classLabelAttribute = ["Cultivar"]
    observationsLabelAttribute = ["Alcohol", "Malic_acid", "Ash", "Alkalinity_ash", "Magnesium", "Phenols", "Flavanoids", "NF_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD", "Proline"]

    df[observationsLabelAttribute] = scaler.fit_transform(df[observationsLabelAttribute].values.astype(float))

    target = df[classLabelAttribute]
    data = df[observationsLabelAttribute]

    train_size = 0.75
    test_size = 0.25
    _shuffle = True;

    train_x, test_x, train_y, test_y = train_test_split(data, 
                                                        target, 
                                                        train_size = train_size, 
                                                        test_size = test_size,
                                                        shuffle = _shuffle)
    
    plt.plot(classifier(train_x, test_x, train_y, test_y), label = 'Wszystkie cechy')
    #plt.show()

    # PCA

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    
    pca_observationsLabelAttribute = ['label 1', 'label 2']
    
    pca_df_temp = pd.DataFrame(data = pca_data, columns = pca_observationsLabelAttribute)
    
    pca_df = pd.concat([pca_df_temp, df[classLabelAttribute]], axis = 1)
        
    pca_target = pca_df[classLabelAttribute]
    pca_data = pca_df[pca_observationsLabelAttribute]
    
    pca_train_x, pca_test_x, pca_train_y, pca_test_y = train_test_split(pca_data, 
                                                                        pca_target, 
                                                                        train_size = train_size, 
                                                                        test_size = test_size,
                                                                        shuffle = _shuffle)

    plt.plot(classifier(pca_train_x, pca_test_x, pca_train_y, pca_test_y), label = 'Wybrane cechy (PCA)')
    #plt.show()

    # wariancja
    
    print(np.var(df))

    var_observationsLabelAttribute = ['Ash', 'Magnesium']

    var_target = pca_df[classLabelAttribute]
    var_data = df[var_observationsLabelAttribute]

    var_train_x, var_test_x, var_train_y, var_test_y = train_test_split(var_data, 
                                                                        var_target, 
                                                                        train_size = train_size, 
                                                                        test_size = test_size,
                                                                        shuffle = _shuffle)

    plt.plot(classifier(var_train_x, var_test_x, var_train_y, var_test_y), label = 'Wybrane cechy (wariancja)')
    #plt.show()

    # chi2

    chi2_data = SelectKBest(chi2, k=2).fit_transform(data, target)

    chi2_observationsLabelAttribute = ['label 1', 'label 2']
    
    chi2_df_temp = pd.DataFrame(data = chi2_data, columns = chi2_observationsLabelAttribute)
    
    chi2_df = pd.concat([chi2_df_temp, df[classLabelAttribute]], axis = 1)
        
    chi2_target = chi2_df[classLabelAttribute]
    chi2_data = chi2_df[pca_observationsLabelAttribute]
    
    chi2_train_x, chi2_test_x, chi2_train_y, chi2_test_y = train_test_split(chi2_data, 
                                                                            chi2_target, 
                                                                            train_size = train_size, 
                                                                            test_size = test_size,
                                                                            shuffle = _shuffle)
    
    plt.plot(classifier(chi2_train_x, chi2_test_x, chi2_train_y, chi2_test_y), label = 'Wybrane cechy (chi2)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()