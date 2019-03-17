from libs import *
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

content = readByColumn("ecoli.data")
names=["SequenceName","mcg","gvh","lip","chg","aac","alm1","alm2"]
#print('\n'.join('{}: {}'.format(*k) for k in enumerate(content)))

for index in range(1,len(names)):
    print("Attribute " + names[index])
    print("\tMedian = " + str(findMedian(content[index])))
    print("\tMax = " + str(findMax(content[index])))
    print("\tMin = " + str(findMin(content[index])))
    #print("\tDominant = " + str(findDominants(content[index])))

print("\nDominanting factor for Class Distribution = " + str(findDominants(content[len(names)])))

print("\nCorrelation check:")
df = pd.DataFrame()
for firstItem in range(1,len(names)):
    df[names[firstItem]] = toFloat(content[firstItem])
print(df.corr())

print("\nDrawing a histogram:")

colors = ['red', 'blue']
labels = ['alm1', 'alm2']
N, bins, patches = plt.hist([toFloat(content[6]),toFloat(content[7])], color=colors, label=labels)
plt.xlabel('ALOM score')
plt.ylabel('Number of samples')
plt.legend(prop={'size': 10})
plt.suptitle('Two most correlated features', fontsize=20)
plt.show()
