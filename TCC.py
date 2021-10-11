from pandas import read_csv
from matplotlib import pyplot

series = read_csv('FT01_18_19.csv', sep=';', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot(figsize=(15,6))
pyplot.legend(['Vazão de Entrada (FT01)'], loc='upper left')
pyplot.xlabel("Time")
pyplot.ylabel("m³/h")
pyplot.show()