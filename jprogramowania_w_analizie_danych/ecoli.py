import re as re
import numpy as np

def readByColumn(filename):
    content = [[]]
    
    with open(filename) as f:
        for x in [re.sub("\s+", ",", res.strip()) for res in f.readlines()]:
            if len(content) == 1:
                content = [[item] for item in x.split(',')]
                continue
            
            for colNum, value in enumerate(x.split(',')):
                content[colNum].append(value)

    return content

content = readByColumn('ecoli.data')
#print('\n'.join('{}: {}'.format(*k) for k in enumerate(content)))
print('median = ' + str(np.median([float(x) for x in content[1]])))
print('min = ' + str(np.min([float(x) for x in content[1]])))
print('max = ' + str(np.max([float(x) for x in content[1]])))

            