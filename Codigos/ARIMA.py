import pandas as pd
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
import pandas.util.testing as tm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from numpy import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Leitura do CSV e setando como Time
df = pd.read_csv('FT01_18_19.csv',delimiter=';', parse_dates=True)
df['Time'] = pd.to_datetime(df.Time)
df = df[df['Time'].notna()]
df = df.set_index('Time')
df = df.resample("D").last()

'''
# Diferenciação de grau um para torná-la estacionária
series_diff = df.diff() 
series_diff.plot()

# Aplicação do Arima
arima = ARIMA(df['Vazao01'], freq='D', order=(5, 1, 2)).fit()
MSE = (arima.resid**2).mean()
print(MSE)
plt.plot(df)
plt.plot(arima.predict(typ='levels'))
plt.show()
'''

'''
# Análise de auto correlação dos lags anteriores
autocorrelation_plot(df)
plot_acf(df, lags=100)


# Executar o ARIMA
arima = ARIMA(df, order=(4,1,2), freq='D')
arima_fit = arima.fit()
print(arima_fit.summary())

# Ver os residuos (erros que aconteceram entre o valor real e o valor predito)
residuals = DataFrame(arima_fit.resid)
residuals.plot()

# Densidade dos Residuos e sua estatistica descritiva 
residuals.plot(kind='kde')
print(residuals.describe())
'''



# ----------------------------------------Prevendo os dados do ARIMA com o WALK FORWARD---------------------------------------------------
X = df.values
X = X.astype('float32')

# Separa 50% dos dados para treino e 50% para teste
size = int(len(X) * 0.50)
train = X[0:size]
test = X[size:]

# Criando variável de previsão e history (history é feita para controle, treina os dados e testa com os dados seguintes)
history = [x for x in train]
predictions = list()

# Cria a função de diferenciação
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# Cria a função que reverte o valor diferenciado para o original
def inverse_difference(history, previsao, interval=1):
    return previsao + history[-interval]






#--------------------------------------Iniciando o Walk Forward--------------------------------------------------
for t in range (len(test)):

    # Difference data
    meses_no_ano = 12
    diff = difference(history, meses_no_ano)

    # Cria um modelo ARIMA com os dados do history
    model = ARIMA(diff, order=(3,0,1))

    # Treina o modelo ARIMA
    model_fit = model.fit(trend='nc', disp=0)

    # A variavel valor_predito recebe o valor previsto pelo modelo
    valor_predito = model_fit.forecast()[0]

    # valor_predito recebe o valor revertido (escala original)
    valor_predito = inverse_difference(history, valor_predito, meses_no_ano)

    # adiciona o valor_predito na lista de predições
    predictions.append(valor_predito)

    # a variavel valor_real recebe o valor real do teste
    valor_real = test[t]

    # adiciona o valor real na variavel history
    history.append(valor_real)

    # imprime valor predito e valor real
    #print('Valor predito=%.3f, Valor esperado=%.3f'%(valor_predito, valor_real))

# Avalia os resultados
RMSE = sqrt(mean_squared_error(test, predictions))
MSE = mean_squared_error(test, predictions)
MAPE = mean_absolute_percentage_error(test, predictions)
print("RMSE é: %.3f" %RMSE)
print("MSE é: %.3f" %MSE)
print("MAPE é: %.3f" %MAPE)

# Plotar os valores reais e valores de predição
plt.plot(test)
plt.plot(predictions, color='red')



# ----------- Analisar os resíduos ------------ pra ver se existem tendencias ou sazonalidades ainda
#Erro residual = Valor esperado - valor predito

# Erros residuais
residuals = [test[i] - predictions[i] for i in range(len(test))]

# Converte a lista em um dataframe
residuals = pd.DataFrame(residuals)

# imprime as 5 primeiras linhas, a descrição e plota
residuals.head()
residuals.describe()
residuals.plot()
plt.show()







