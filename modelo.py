from sklearn.preprocessing import PolynomialFeatures  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def regressão_polinomial(X, y):
    conversor_polinomial = PolynomialFeatures(degree=2, include_bias=False)
    X_poli = conversor_polinomial.fit_transform(X)

    X_treino, X_teste, y_treino, y_teste = train_test_split(X_poli, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_treino,y_treino)
    previsoes = modelo.predict(X_teste)

    erro_medio_absoluto = mean_absolute_error(previsoes, y_teste)
    raiz_do_erro_medio_quadratico = root_mean_squared_error(previsoes, y_teste)

    # Fazendo estimativas com dados informados pelo usuário

    dados_usuario = []

    try:
        while len(dados_usuario) < 3:    
        #Recebendo a Área do usuário
            area = float(input('Digite aqui a área da casa: '))
            dados_usuario.append(area)

        #Recebendo quantidade de quartos do usuário
            quartos = float(input('Digite aqui quantos quartos tem na casa: '))
            dados_usuario.append(quartos)

        #Recebendo idade da casa do usuário
            idade = float(input('Digite aqui quantos anos a casa tem: '))
            dados_usuario.append(idade)

    except ValueError:
        print('O valor digitado é inválido! Por favor, digite um valor válido.')

    #Calculando preço baseado nos dados fornecidos pelo usuário   

    dados_transformados = conversor_polinomial.transform([dados_usuario])
    previsao = modelo.predict(dados_transformados).round(2)

    print('---------------------------------------------------------------')
    print(f'Sua casa tem o preço de mercado de aproximadamente R${previsao[0]}')
    print('---------------------------------------------------------------')

    # print(f"O modelo fez as previsões e as taxas de erro são a média de: \n\nMAE: {erro_medio_absoluto}; \nRMSE: {raiz_do_erro_medio_quadratico}")
