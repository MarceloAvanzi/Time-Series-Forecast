import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Ler o CSV
df = pd.read_csv('Dados_Sanepar.csv', sep=';')
print(df.head())

#Mudar o Type da coluna Time e colocar como index para plotar
df.Time = pd.to_datetime(df.Time)
df.set_index('Time', inplace=True)
df.info()

#Plotar as três vazões
#df.plot(figsize=(15,6))
#plt.show()

#Aplicar média movel para identificar a tendencia
#df.ft01.rolling(12).mean().plot(figsize=(15,6))
#plt.show()

#Outra maneira para achar a tendencia
#df.ft01.groupby(df.index.year).sum().plot(figsize=(15,6))
#plt.show()

#Analisando a sazonalidade (removendo as tendencias, fazendo o z(t) = z(t) - z(t-1), e aplicando filtro para facilitar a vista
#filter = (df.index.year >= 2019) & (df.index.year <= 2019)
#df[filter].ft01.diff().plot(figsize=(15,6))
#plt.show()

#outra forma de analisar a sazonalidade (agrupando por mes)
#df.ft01.diff().groupby(df.index.month).mean().plot(kind='bar')
#plt.show()





