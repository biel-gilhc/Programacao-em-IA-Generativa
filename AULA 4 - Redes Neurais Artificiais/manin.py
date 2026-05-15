import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURAÇÃO DA INTERFACE (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Preditor de Notas", layout="centered")
st.title("🎓 Predição de Notas por Horas de Estudo")
st.markdown(
    "Demonstração didática de uma **Regressão Linear Simples** utilizando "
    "`scikit-learn` para modelagem e `Matplotlib` para representação gráfica."
)

# ==========================================
# 2. CAMADA DE DADOS E MODELAGEM (IA)
# ==========================================
# Conjunto de dados fornecido para o exercício
estudos = pd.DataFrame(
    {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
)

# Adequação dimensional: X precisa ser uma matriz (2D) e y um vetor (1D)
X = estudos[["horas"]]
y = estudos["notas"]

# Instanciação e ajuste do estimador linear
modelo = LinearRegression()
modelo.fit(X, y)

# ==========================================
# 3. INTERAÇÃO E INFERÊNCIA EM TEMPO REAL
# ==========================================
st.sidebar.header("Parâmetros do Estudante")
horas_digitadas = st.sidebar.slider(
    "Quantidade de horas estudadas:",
    min_value=0.0,
    max_value=12.0,
    value=6.0,
    step=0.5,
)

# Realização da predição (a entrada deve ser estritamente bidimensional)
entrada_usuario = np.array([[horas_digitadas]])
nota_predita = modelo.predict(entrada_usuario)[0]

# Tratamento de contorno: Restringir a nota aos limites acadêmicos padrão (0 a 10)
nota_final = max(0.0, min(10.0, nota_predita))

# Exibição dos resultados em componentes de métrica
col1, col2 = st.columns(2)
col1.metric(label="Tempo de Estudo", value=f"{horas_digitadas} horas")
col2.metric(label="Nota Estimada", value=f"{nota_final:.2f} / 10")

# ==========================================
# 4. REPRESENTAÇÃO GRÁFICA (VISUALIZAÇÃO)
# ==========================================
st.subheader("Análise Visual do Ajuste da Reta")

# Geração de pontos equidistantes para plotar a reta teórica de regressão
X_reta = np.linspace(0, 12, 100).reshape(-1, 1)
y_reta = np.clip(modelo.predict(X_reta), 0, 10)

# Construção do gráfico com Matplotlib
fig, ax = plt.subplots(figsize=(8, 4.5))

# Plot dos dados históricos conhecidos (Scatter Plot)
ax.scatter(
    estudos["horas"],
    estudos["notas"],
    color="#005088",
    s=80,
    label="Dados Históricos",
    zorder=3,
)

# Plot da reta matemática gerada pelo modelo (Line Plot)
ax.plot(
    X_reta,
    y_reta,
    color="#f44336",
    linestyle="--",
    linewidth=2,
    label="Reta de Regressão",
)

# Destaque visual do ponto atual simulado pelo usuário
ax.scatter(
    horas_digitadas,
    nota_final,
    color="#11caa0",
    s=150,
    marker="X",
    label="Sua Predição",
    zorder=4,
)

# Customização técnica do plano cartesiano
ax.set_xlabel("Horas de Estudo")
ax.set_ylabel("Nota Obtida")
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left")

# Renderização do componente gráfico dentro do Streamlit
st.pyplot(fig)

# ==========================================
# 5. METADADOS E EQUAÇÃO MATEMÁTICA
# ==========================================
with st.expander("Expandir para detalhes matemáticos do modelo"):
    st.markdown("A reta foi calculada através do método dos Mínimos Quadrados:")
    st.latex(rf"f(x) = {modelo.coef_[0]:.2f}x + ({modelo.intercept_:.2f})")
    st.text(f"Intercepto (b): {modelo.intercept_:.4f}")
    st.text(f"Coeficiente Angular (m): {modelo.coef_[0]:.4f}")