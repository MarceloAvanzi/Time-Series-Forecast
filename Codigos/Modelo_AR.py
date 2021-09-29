import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR

#Ler o CSV
df = pd.read_csv('Teste.csv', sep=';')
print(df.head())

#Mudar o Type da coluna Time e colocar como index para plotar
df.Time = pd.to_datetime(df.Time)
df.set_index('Time', inplace=True)

#Criar variavel de input pro modelo AR
df['x1'] = df.Vazao01.shift(1)
df.dropna(axis=0, inplace=True) #Elimina primeira linha que tinha NaN
print(df.head())

#Modelo de Regressão
X = df.x1.values
y = df.Vazao01.values

reg = LinearRegression().fit(X.reshape(-1, 1), y)
previsoes = reg.predict(X.reshape(-1, 1))

#Calculo do MSE
erro = (y - previsoes) ** 2
MSE = erro.mean()
print(MSE)

#plotar
plt.scatter(X, y)
plt.plot(X, reg.predict(X.reshape(-1, 1)), color='red')
plt.xlabel('x1')
plt.ylabel('Vazao01')
#plt.show()

# AR
df['ar_1'] = previsoes
df[['Vazao01', 'ar_1']].plot()
plt.show()

