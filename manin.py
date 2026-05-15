# 2

import pandas as pd
from sklearn.linear_model import LinearRegression

# Criando os dados
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# Variável de entrada
X = gamer[['horas_jogo']]

# Variável alvo
y = gamer['cansaco']

# Criando o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, y)

# Fazendo previsão
horas = 7
previsao = modelo.predict([[horas]])

# Resultados
print("Coeficiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)
print(f"Previsão de cansaço para {horas} horas jogando: {previsao[0]:.2f}")


#####################

#3

# IA do Sorvete
# Objetivo: prever a quantidade de sorvetes vendidos pela temperatura

import pandas as pd
from sklearn.linear_model import LinearRegression

# Dados
sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# Variável independente (X)
X = sorvete[['temperatura']]

# Variável dependente (y)
y = sorvete['vendas']

# Criando o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, y)

# Temperatura para previsão
nova_temperatura = [[32]]

# Fazendo previsão
previsao = modelo.predict(nova_temperatura)

# Resultado
print("Temperatura:", nova_temperatura[0][0], "°C")
print("Previsão de vendas:", round(previsao[0], 2), "sorvetes")


###############

#4
# Detector de Aprovação Ninja
# Objetivo: classificar aluno como aprovado ou reprovado

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dados
alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]
})

# Variável independente (X)
X = alunos[['faltas']]

# Variável dependente (y)
y = alunos['resultado']

# Criando o modelo
modelo = LogisticRegression()

# Treinando o modelo
modelo.fit(X, y)

# Quantidade de faltas para teste
novas_faltas = [[3]]

# Fazendo previsão
previsao = modelo.predict(novas_faltas)

# Resultado
if previsao[0] == 1:
    print("Aluno aprovado")
else:
    print("Aluno reprovado")

    #######################


#5
    # IA do Pet Feliz
# Objetivo: prever felicidade do cachorro

import pandas as pd
from sklearn.linear_model import LinearRegression

# Dados
pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

# Variável independente (X)
X = pets[['passeios']]

# Variável dependente (y)
y = pets['felicidade']

# Criando o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, y)

# Quantidade de passeios para previsão
novos_passeios = [[6]]

# Fazendo previsão
previsao = modelo.predict(novos_passeios)

# Resultado
print("Passeios por dia:", novos_passeios[0][0])
print("Previsão de felicidade:", round(previsao[0], 2))