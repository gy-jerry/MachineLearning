'''
scripts using kNN.py

'''

import kNN

group, labels = kNN.createDataSet()
print('group = ', group)
print('labels = ', labels)

classify_results = kNN.classify0([0, 0], group, labels, 3)
print('classify_results:', classify_results)