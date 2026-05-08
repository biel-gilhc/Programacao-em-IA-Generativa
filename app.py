import streamlit as st
import pandas as pd

st.title('teste')
pd.read_csv('vendas.csv')
dados = pd.read_csv('vendas.csv')


df = pd.DataFrame(dados)
st.bar_chart(df , x = 'nome', y = 'venda')

#------------------------------------------------------
st.caption('paragrafo')
st.audio('audio.mp3')
st.image('img.jpg', width= 250)

#-------------------------------------------------------

# #calculadora 
# st.title('calculadora')
# st.caption('colocar 1 depois outro numero escolher a operaçao e ver o resultado')

# n1 = st.number_input('primeiro numero')
# n2 = st.number_input('segundo numero')

# escolha = st.selectbox('escolha a operaçao', ['+', '-', '*', '/'])

# if st.button('calcular'):
#    if escolha == '+':
#     soma = n1 + n2
#     resultado = soma 
#    elif escolha == '-':
#      sub = n1 - n2
#      resultado = sub
#    elif escolha == '*':
#      mult = n1 * n2
#      resultado = mult
#    elif escolha == '/':
#      div = n1 / n2
#      resultado = div 
    
#    st.success(resultado)
#-----------------------------------------------------------------------

st.title('imc')

n3 = st.number_input('peso')

n4 = st.number_input('altura', value=1.0)

imc = n3/n4**2
if st.button('calcular'):
   if imc:
   
     st.success(imc)

#----------------------------------------------------------------

s = 1234
u = 'ana'

usuario = (input('crie um nome de usuario'))
senha = int(input('crie uma senha:'))
if u == usuario and senha == s:
   print('acesso autorizadpo')
else:
   print('acesso negsdo')   
