import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR
from datetime import datetime

#Ler o CSV
df = pd.read_csv('teste.csv', encoding = "UTF-8", sep=';')
#print(df)

#Mudar o Type da coluna Time e colocar como index para plotar
df.Time = pd.to_datetime(df.Time)
df.set_index('Time', inplace=True)

#Usando statsmodels

ar_2 = AR(df.Vazao01, freq='MS').fit(2)
#ar_2.resid