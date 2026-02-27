from modelo import regressao_polinomial
from graficos import g_real_vs_predito, g_residuos, histograma_residuos

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np

#importando e limpando o DataFrame

df_residencias = pd.read_excel('/home/davi-dos-santos-freire/VSCode/RP_imóveis/residencias.xlsx')

#Limpando o DataFrame

df_residencias['area_m2'] = pd.to_numeric(df_residencias['area_m2'], errors='coerce')
df_residencias['quartos'] = pd.to_numeric(df_residencias['quartos'], errors='coerce')           #Transformando valores que não numéricos em NaN
df_residencias['idade_casa'] = pd.to_numeric(df_residencias['idade_casa'], errors='coerce')
df_residencias['preco'] = pd.to_numeric(df_residencias['preco'], errors='coerce')

numeric_cols = df_residencias.select_dtypes(include=['number']).columns
df_residencias = df_residencias[(df_residencias[numeric_cols] >= 0).all(axis=1)]                #Removendo valores negativos do DataFrame

df_residencias = df_residencias.dropna()                                                        #Removendo linhas com valores NaN do DataFrame

df_residencias.head()
#Definindo variáveis

X = df_residencias.drop('preco', axis=1)
y = df_residencias['preco']

#usando o modelo de regressão polinomial no DataFrame

regressao_polinomial(X, y)

#Criando gráficos do modelo