import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

def resultado_copa():
    from manin import gols_input


st.title('ENSINA A MÁQUINA PREVER O FUTURO')
st.write('Preve o campeão da copa')

st.header('opções de campeão...')

# dados 
dados = pd.DataFrame ({
'gols':[12,15,10,18,14,11,16],
'ranking':[1,3,2,1,4,10,2],
'pais':['Brasil','Argentina' ,'França' ,'Brasil' ,'França' ,'Argentina', 'Brasil']
})
# alinhamento do modelo
modelo_copa = DecisionTreeClassifier()
# treinamento
modelo_copa.fit(dados[['gols', 'ranking']], dados['pais'])

gols_input = st.number_input('Quantos gol o time fez?', 0,30,15)
rank_input =st.number_input('Qual posição', 1,100,1)

if st.button('Prever'):
    #previsão
    resultado_copa = modelo_copa.predict([[gols_input, rank_input]])
    st.success(F'o provavel campeão é{resultado_copa}')

# config.py

gols_input = ...
modelo_copa = ...
rank_input = ...

# ___________________________________________



# NOTAS DE ESTUDOS 



st.header('ANALISE DE NOTAS - PREVENDO')
estudos = pd.DataFrame({
'notas':[1,2,4,6,8,10],
'horas':[2,4,5,7,9,10]
})
