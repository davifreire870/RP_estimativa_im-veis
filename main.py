from modelo import Regr_polinomial

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np

#importando e limpando o DataFrame

df_resid = pd.read_excel('/home/davi-dos-santos-freire/VSCode/RP_imóveis/residencias.xlsx')

#Limpando o DataFrame

df_resid['area_m2'] = pd.to_numeric(df_resid['area_m2'], errors='coerce')
df_resid['quartos'] = pd.to_numeric(df_resid['quartos'], errors='coerce')           #Transformando valores que não numéricos em NaN
df_resid['idade_casa'] = pd.to_numeric(df_resid['idade_casa'], errors='coerce')
df_resid['preco'] = pd.to_numeric(df_resid['preco'], errors='coerce')

df_resid = df_resid.dropna()                                                        #Removendo linhas com valores NaN do DataFrame

#Definindo variáveis

X = df_resid.drop('preco', axis=1)
y = df_resid['preco']

#usando o modelo de regressão polinomial no DataFrame

Regr_polinomial(X, y)

