#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


#Ler arquivo e converter coluna mes para tipo data, deixando ela como index da base 

dateparse = lambda dates : pd.datetime.strptime(dates, '%Y-%m')
base = pd.read_csv('AirPassengers.csv', parse_dates = ['Month'],
                    index_col = 'Month', date_parser = dateparse)


# In[4]:


#Mudando o tipo para Series 
ts=base['#Passengers']
ts


# In[6]:


plt.plot(ts)


# In[9]:


#p numero de termos alto regressivos
#q é o numero da média móvel
#d é a diferença não sazonais
modelo = ARIMA(ts,order=[2,1,2])
modelo_treinado = modelo.fit()


# In[11]:


modelo_treinado.summary()


# In[12]:


#steps = quandidade de previsoes que quer fazer 
previsoes = modelo_treinado.forecast(steps=12)[0]
print(previsoes)


# In[22]:


eixo = ts.plot()
#aonde começa os dados que serão usados para prever até o final do valor da previsão
modelo_treinado.plot_predict('1960-01-01', '1962-01-01',
                                ax = eixo,plot_insample=False)


# In[ ]:




