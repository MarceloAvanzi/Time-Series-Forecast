from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot

series = read_csv('FT01_18_19.csv', sep=';', header=0, index_col=0, parse_dates=True, squeeze=True)

# Plotagem normal da Vazão
series.plot(figsize=(14,6))
pyplot.title("Plot Normal FT01")
pyplot.legend(['Vazão de Entrada (FT01)','Vazão de Gravidade (FT02)','Vazão de Recalque (FT03)'], loc='upper left')
pyplot.xlabel("Time")
pyplot.ylabel("m³/h")
pyplot.figure()

# Plotagem do Histograma
series.hist(figsize=(10,5))
pyplot.title("Plot Histograma FT01")
pyplot.legend(['Vazão de Entrada (FT01)','Vazão de Gravidade (FT02)','Vazão de Recalque (FT03)'], loc='upper left')
pyplot.xlabel("Time")
pyplot.ylabel("m³/h")
pyplot.figure()

# Plotagem de densidade
series.plot(kind='kde')
pyplot.legend(['Vazão de Entrada (FT01)','Vazão de Gravidade (FT02)','Vazão de Recalque (FT03)'], loc='upper left')
pyplot.show()