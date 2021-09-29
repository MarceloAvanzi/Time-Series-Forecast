from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from pandas.plotting import lag_plot
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose

path = 'https://github.com/MarceloAvanzii/TCC/blob/main/teste.csv'
series = read_csv(path, sep=';', header=0, index_col=0, parse_dates=True, squeeze=True)
#print(series['2018-01'])


#modelo Naive
'''
dataframe = DataFrame()
dataframe['year'] = [series.index[i].year for i in range(len(series))]
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['vazao'] = [series[i] for i in range(len(series))]
print(dataframe.head(5))
'''


#Lag de t+1
'''
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t', 't+1']
print(dataframe.head(5))
'''

#Lag de t-2 t-1 e t+1
'''
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-2', 't-1', 't', 't+1']
print(dataframe.head(5))
'''

#Rolling mean
'''
temps = DataFrame(series.values)
shifted = temps.shift(1)
window = shifted.rolling(window=2)
means = window.mean()
dataframe = concat([means, temps], axis=1)
dataframe.columns = ['mean(t-1,t)', 't+1']
print(dataframe.head(5))
'''

#Rolling mean com 5
'''
temps = DataFrame(series.values)
width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(window=width)
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
'''

#Expanding window
'''
temps = DataFrame(series.values)
window = temps.expanding()
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))
'''

#Line Plot or dot plot
'''
series.plot(style='k.')
pyplot.show()
'''

#Comparing plots by year
'''
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.plot(subplots=True, legend=True)
pyplot.show()
'''

#Plotando histograma
'''
series.hist()
pyplot.show()
'''

# Plotando para saber a densidade
'''
series.plot(kind='kde')
pyplot.show()
'''

#Plot de box e whisker
'''
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
pyplot.show()
'''

#Plot de box e whisker separado por mes em 2018
'''
one_year = series['2018']
groups = one_year.groupby(Grouper(freq='M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()
'''

#heat map
'''
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()
'''

#heat map separado por mes em 2018
'''
one_year = series['2018']
groups = one_year.groupby(Grouper(freq='M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()
'''

#lag plot
'''
lag_plot(series)
pyplot.show()
'''

#lag plot comparando com os dias atras
'''
values = DataFrame(series.values)
lags = 7
columns = [values]
for i in range(1,(lags + 1)):
    columns.append(values.shift(i))
dataframe = concat(columns, axis=1)
columns = ['t']
for i in range(1,(lags + 1)):
    columns.append('t-' + str(i))
dataframe.columns = columns
pyplot.figure(1)
for i in range(1, (lags + 1)):
    ax = pyplot.subplot(240 + i)
    ax.set_title('t vs t-' + str(i))
    pyplot.scatter(x=dataframe['t'].values, y=dataframe['t-' + str(i)].values)
pyplot.show()
'''

#auto correlação entre observação e lag
'''
autocorrelation_plot(series)
pyplot.show()
'''


#decomposição em sazonalidade tendencia e residuo etc
result = seasonal_decompose(series, model='additive',freq=30)
result.plot()
pyplot.show()



















































