import configparser
import pandas as pd

from test_null_trained_auto import null_trained_auto
from test_hurst_trained_auto import hurst_trained_auto
from test_dfa_trained_auto import dfa_trained_auto

from Transaction_model_ProfitAndLossChanges_2 import profitAndLossChanges

config = configparser.ConfigParser()
config.read('config.ini', encoding='GB18030')

config.set('config', 'count', '0')
config.set('config', 'filename', 'pred_null.csv')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
series = pd.Series()
for count in range(1, 31):
    null_trained_auto()
    series = profitAndLossChanges()
    series.to_csv('ProfitAndLossChanges/ProfitAndLossChanges_null_'+count.__str__()+'.csv', index=False)

config.set('config', 'count', '0')
config.set('config', 'filename', 'pred_hurst.csv')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
series = pd.Series()
for count in range(1, 31):
    hurst_trained_auto()
    series = profitAndLossChanges()
    series.to_csv('ProfitAndLossChanges/ProfitAndLossChanges_hurst_' + count.__str__() + '.csv', index=False)

config.set('config', 'count', '0')
config.set('config', 'filename', 'pred_dfa.csv')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
series = pd.Series()
for count in range(1, 31):
    dfa_trained_auto()
    series = profitAndLossChanges()
    series.to_csv('ProfitAndLossChanges/ProfitAndLossChanges_dfa_' + count.__str__() + '.csv', index=False)