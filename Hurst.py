from __future__ import division
import pandas as pd
from numpy import std, subtract, polyfit, sqrt, log

from collections import Iterable
import numpy as np
from pandas import Series
from hurst import compute_Hc


train_data = pd.read_csv('5m.csv', encoding='gbk')
close = train_data['close']
shape = close.shape[0]
print(shape)

closegroup = []
for i in range(1,707):
    for j in range(1,7):
        closegroup.append(close[(i-1)*48+j-1])

#print(closegroup)

countgroup = 668
hrustgroup = []
hurstline = []
for i in range(1,countgroup):
    hurst_exponent, c, data = compute_Hc(closegroup[(i-1)*6:(i-1)*6+240], kind='price')
    hrustgroup.append(hurst_exponent)
    for j in range(1, 49):
        hurstline.append(hurst_exponent)
    print(i)

for i in range(len(hurstline), 33936):
    hurstline.append('')
allDataFrame = pd.read_csv('5m.csv')
allDataFrame['hurst'] = hurstline
allDataFrame.to_csv('7m.csv', index=False)

import matplotlib.pyplot as plt
plt.plot(range(1,countgroup),hrustgroup, 'b', label='Hrust')
plt.legend()
plt.show()
