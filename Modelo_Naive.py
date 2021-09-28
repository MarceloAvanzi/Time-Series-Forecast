import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Ler o CSV
df = pd.read_csv('Dados_Vazão1.csv', sep=';')
print(df.head())

#Mudar o Type da coluna Time e colocar como index para plotar
df.Time = pd.to_datetime(df.Time)
df.set_index('Time', inplace=True)

#Previsão Naive
df['Naive'] = df.Vazao01.shift(1)
print(df.head())
df.plot(figsize=(15,6))
plt.show()

#Validação (MSE)
erro = (df.Vazao01 - df.Naive) ** 2
erro.mean()

