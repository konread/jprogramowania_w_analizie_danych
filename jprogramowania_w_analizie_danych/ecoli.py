from libs import *

content = readByColumn('ecoli.data')
names=['SequenceName','mcg','gvh','lip','chg','aac','alm1','alm2']
#print('\n'.join('{}: {}'.format(*k) for k in enumerate(content)))

for index in range(1,len(names)):
    print('Attribute ' + names[index])
    print('\tMedian = ' + str(findMedian(content[index])))
    print('\tMax = ' + str(findMax(content[index])))
    print('\tMin = ' + str(findMin(content[index])))
    print('\tDominant = ' + str(findDominants(content[index])))

