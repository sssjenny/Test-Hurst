import configparser
import pandas as pd

from test_null_trained_auto import null_trained_auto
from test_hurst_trained_auto import hurst_trained_auto
from test_dfa_trained_auto import dfa_trained_auto

config = configparser.ConfigParser()
config.read('config.ini', encoding='GB18030')

config.set('config', 'count', '0')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
r_square = 0
for count in range(1, 31):
    r_square = null_trained_auto()
    auto_result[count.__str__()] = [r_square]
auto_result.to_csv('auto_trained_result_null.csv', encoding='utf-8', index=False)

config.set('config', 'count', '0')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
r_square = 0
for count in range(1, 31):
    r_square = hurst_trained_auto()
    auto_result[count.__str__()] = [r_square]
auto_result.to_csv('auto_trained_result_hurst.csv', encoding='utf-8', index=False)

config.set('config', 'count', '0')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
r_square = 0
for count in range(1, 31):
    r_square = dfa_trained_auto()
    auto_result[count.__str__()] = [r_square]
auto_result.to_csv('auto_trained_result_dfa.csv', encoding='utf-8', index=False)