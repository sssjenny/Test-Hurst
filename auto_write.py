import configparser
import pandas as pd

from Transaction_model_X_auto import cal_X_auto
from Transaction_model_AnnualizedRateOfReturn import annualizedRateOfReturn
from Transaction_model_Drawdown import drawdown
from Transaction_model_SharpeRatio import sharpeRatio
from Transaction_model_SortinoRatio import sortinoRatio
from Transaction_model_Profit_lossRatio import profit_lossRatio
from Transaction_model_WinRate import winRate
from test_null_auto import null_auto
from test_hurst_auto import hurst_auto
from test_dfa_auto import dfa_auto

config = configparser.ConfigParser()
config.read('config.ini', encoding='GB18030')

config.set('config', 'count', '10')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
for count in range(1, 21):
    mse, mae = null_auto()
    maxReturn = cal_X_auto()
    aunRateOfReturn = annualizedRateOfReturn()
    maxDrawdown = drawdown()
    thissharpeRatio = sharpeRatio()
    thissortinoRatio = sortinoRatio()
    thisprofit_lossRatio = profit_lossRatio()
    thiswinRate = winRate()
    auto_result[count.__str__()] = [maxReturn, aunRateOfReturn, maxDrawdown,
                                thissharpeRatio, thissortinoRatio, thisprofit_lossRatio,
                                thiswinRate, mse, mae]

auto_result.to_csv('auto_result_null.csv', encoding='utf-8', index=False)


config.set('config', 'count', '10')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
for count in range(1, 21):
    mse, mae = hurst_auto()
    maxReturn = cal_X_auto()
    aunRateOfReturn = annualizedRateOfReturn()
    maxDrawdown = drawdown()
    thissharpeRatio = sharpeRatio()
    thissortinoRatio = sortinoRatio()
    thisprofit_lossRatio = profit_lossRatio()
    thiswinRate = winRate()
    auto_result[count.__str__()] = [maxReturn, aunRateOfReturn, maxDrawdown,
                                thissharpeRatio, thissortinoRatio, thisprofit_lossRatio,
                                thiswinRate, mse, mae]

auto_result.to_csv('auto_result_hurst.csv', encoding='utf-8', index=False)


config.set('config', 'count', '10')
openConfig = open('config.ini', 'w')
config.write(openConfig)
openConfig.close()
auto_result = pd.DataFrame()
for count in range(1, 21):
    mse, mae = dfa_auto()
    maxReturn = cal_X_auto()
    aunRateOfReturn = annualizedRateOfReturn()
    maxDrawdown = drawdown()
    thissharpeRatio = sharpeRatio()
    thissortinoRatio = sortinoRatio()
    thisprofit_lossRatio = profit_lossRatio()
    thiswinRate = winRate()
    auto_result[count.__str__()] = [maxReturn, aunRateOfReturn, maxDrawdown,
                                thissharpeRatio, thissortinoRatio, thisprofit_lossRatio,
                                thiswinRate, mse, mae]

auto_result.to_csv('auto_result_dfa.csv', encoding='utf-8', index=False)