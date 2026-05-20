"""
╔══════════════════════════════════════════════════════════╗
║   💸 Projetor de Conta de Luz — Ar-Condicionado + IA     ║
║   Stack: Streamlit + TensorFlow (Keras Linear Model)     ║
╚══════════════════════════════════════════════════════════╝

CONCEITO DIDÁTICO:
-----------------
Poderíamos calcular o custo com uma fórmula direta.
Mas aqui usamos TensorFlow para APRENDER essa relação
a partir de dados — como um modelo real de ML faz.

Fluxo do app:
  1. Geramos dados de treino simulando consumo real
  2. Treinamos uma rede neural linear (1 neurônio)
  3. O modelo aprende: "horas → custo em R$"
  4. Usamos o modelo treinado para prever os próximos 30 dias
  5. Exibimos o gráfico de evolução do gasto acumulado
"""

# ─── IMPORTAÇÕES ──────────────────────────────────────────
import streamlit as st          # Framework de interface web
import numpy as np              # Operações numéricas/arrays
import pandas as pd             # Estrutura de dados (para st.line_chart)
import tensorflow as tf         # Machine Learning / redes neurais
from tensorflow import keras    # API de alto nível do TensorFlow


# ─── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────
st.set_page_config(
    page_title="Projetor de Conta de Luz",
    page_icon="❄️",
    layout="centered"
)


# ─── CONSTANTES DO DOMÍNIO ────────────────────────────────
# Esses valores representam um ar-condicionado split 9.000 BTUs típico
POTENCIA_WATTS = 1200        # Consumo médio em Watts
TARIFA_KWH = 0.95            # R$ por kWh (tarifa média Brasil 2024)
DIAS_PROJECAO = 30           # Quantos dias vamos projetar


# ─── FUNÇÃO: GERAR DADOS DE TREINO ────────────────────────
@st.cache_data  # Cacheia para não regenerar a cada interação
def gerar_dados_treino():
    """
    Simula um histórico de uso do ar-condicionado.

    Criamos 200 amostras com horas variando de 0 a 12h/dia,
    e calculamos o custo com um leve ruído (noise) para
    imitar variações reais de tensão, temperatura, etc.

    Retorna:
        X: array de horas (entrada do modelo)
        y: array de custos em R$ (saída esperada)
    """
    np.random.seed(42)  # Semente para reprodutibilidade

    # Horas de uso: valores aleatórios entre 0 e 12
    X = np.random.uniform(0, 12, size=200).astype(np.float32)

    # Custo real = potência × horas / 1000 × tarifa
    # + ruído aleatório pequeno para simular variações reais
    ruido = np.random.normal(0, 0.05, size=200).astype(np.float32)
    y = (POTENCIA_WATTS * X / 1000 * TARIFA_KWH + ruido).astype(np.float32)

    return X, y


# ─── FUNÇÃO: CONSTRUIR E TREINAR O MODELO ─────────────────
@st.cache_resource  # Cacheia o modelo (objeto TF não é serializável com cache_data)
def treinar_modelo():
    """
    Cria e treina uma rede neural minimalista com Keras.

    Arquitetura:
        Entrada (1 feature: horas) → Dense(1) → Saída (custo R$)

    É basicamente uma regressão linear aprendida pelo gradiente.
    O modelo aprende os parâmetros: custo ≈ W * horas + b
    onde W (peso) e b (bias) são ajustados automaticamente.

    Retorna:
        model: modelo Keras treinado
        historico: métricas de treino (loss por época)
    """
    X_treino, y_treino = gerar_dados_treino()

    # --- Construção da arquitetura ---
    model = keras.Sequential([
        # Dense(1): 1 neurônio, sem ativação = regressão linear
        # input_shape=(1,) porque temos 1 feature: horas
        keras.layers.Dense(units=1, input_shape=(1,))
    ])

    # --- Compilação: define otimizador e função de perda ---
    # mean_squared_error = MSE, clássico para regressão
    # adam = otimizador adaptativo (ajusta taxa de aprendizado)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mean_squared_error"
    )

    # --- Treino: 300 épocas, verbose=0 = sem logs no terminal ---
    historico = model.fit(
        X_treino,
        y_treino,
        epochs=300,
        verbose=0,
        batch_size=32  # Processa 32 amostras por vez (mini-batch)
    )

    return model, historico


# ─── INTERFACE: CABEÇALHO ─────────────────────────────────
st.title("❄️ Projetor de Conta de Luz")
st.subheader("Ar-condicionado com previsão por IA (TensorFlow)")

st.markdown("""
> **Como funciona?** Um modelo de Machine Learning foi treinado com dados
> simulados de consumo. Ele aprendeu a relação entre *horas de uso* e *custo*.
> Agora usamos esse modelo para **projetar seus gastos nos próximos 30 dias**.
""")

st.divider()


# ─── INTERFACE: INPUTS DO USUÁRIO ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    # Slider principal: quantas horas por dia o usuário usa o AC
    horas_uso = st.slider(
        label="🕐 Horas de uso por dia",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
        help="Arraste para ajustar quantas horas/dia você usa o ar-condicionado"
    )

with col2:
    # Permite o usuário ajustar a tarifa (varia por estado/concessionária)
    tarifa_usuario = st.number_input(
        label="💰 Tarifa kWh (R$)",
        min_value=0.50,
        max_value=2.00,
        value=TARIFA_KWH,
        step=0.05,
        format="%.2f",
        help="Consulte sua conta de luz. Média nacional ~R$ 0,95"
    )

st.divider()


# ─── TREINAMENTO COM FEEDBACK VISUAL ──────────────────────
with st.spinner("🧠 Treinando modelo de IA... (só na primeira vez)"):
    modelo, historico_treino = treinar_modelo()

st.success("✅ Modelo treinado com sucesso!")


# ─── PREVISÃO: PROJEÇÃO DOS 30 DIAS ───────────────────────
# Criamos um array com as horas de uso para cada dia
# Como o usuário usa X horas/dia, repetimos X por 30 dias
horas_array = np.full(
    shape=(DIAS_PROJECAO,),
    fill_value=horas_uso,
    dtype=np.float32
)

# O modelo prevê o custo DIÁRIO para cada entrada de horas
# predict() retorna shape (30, 1) → flatten() vira (30,)
custos_diarios = modelo.predict(horas_array, verbose=0).flatten()

# Calculamos o custo acumulado: cada dia soma o anterior
# Ex: [2.0, 2.0, 2.0] → [2.0, 4.0, 6.0]
custos_acumulados = np.cumsum(custos_diarios)

# Ajuste proporcional: o modelo foi treinado com tarifa padrão,
# então escalamos para a tarifa informada pelo usuário
fator_ajuste = tarifa_usuario / TARIFA_KWH
custos_acumulados_ajustados = custos_acumulados * fator_ajuste
custos_diarios_ajustados = custos_diarios * fator_ajuste


# ─── MÉTRICAS DE DESTAQUE ─────────────────────────────────
m1, m2, m3 = st.columns(3)

custo_hoje = float(custos_diarios_ajustados[0])
custo_semana = float(custos_acumulados_ajustados[6])   # dia 7
custo_mes = float(custos_acumulados_ajustados[-1])     # dia 30

m1.metric(
    label="💡 Custo hoje",
    value=f"R$ {custo_hoje:.2f}"
)
m2.metric(
    label="📅 Projeção 7 dias",
    value=f"R$ {custo_semana:.2f}"
)
m3.metric(
    label="🗓️ Projeção 30 dias",
    value=f"R$ {custo_mes:.2f}",
    delta=f"+R$ {custo_mes:.2f} na conta"
)


# ─── GRÁFICO DE LINHA: EVOLUÇÃO DO GASTO ──────────────────
st.subheader("📈 Evolução do gasto acumulado (30 dias)")

# Montamos um DataFrame para o st.line_chart
# Índice = dia (1 a 30), coluna = valor acumulado
df_grafico = pd.DataFrame(
    data={
        "Gasto Acumulado (R$)": custos_acumulados_ajustados,
        "Gasto Diário (R$)": custos_diarios_ajustados,
    },
    index=pd.RangeIndex(start=1, stop=DIAS_PROJECAO + 1, name="Dia")
)

# st.line_chart renderiza automaticamente com zoom e tooltip
st.line_chart(df_grafico, use_container_width=True)


# ─── CONTEXTO EDUCATIVO ───────────────────────────────────
with st.expander("🔬 O que o modelo aprendeu? (detalhes técnicos)"):
    # Extraímos os pesos aprendidos pelo neurônio
    peso_w = modelo.layers[0].get_weights()[0][0][0]  # W (inclinação)
    bias_b = modelo.layers[0].get_weights()[1][0]      # b (intercepto)

    st.markdown(f"""
    O modelo Keras aprendeu uma **equação linear**:

    ```
    custo (R$) ≈ {peso_w:.4f} × horas + {bias_b:.4f}
    ```

    Compare com a fórmula real:
    ```
    custo = {POTENCIA_WATTS}W × horas ÷ 1000 × R${TARIFA_KWH}
          = {POTENCIA_WATTS/1000 * TARIFA_KWH:.4f} × horas
    ```

    **O modelo chegou muito próximo do valor real sem conhecer a fórmula!**
    Ele aprendeu apenas observando os dados de exemplo. Isso é Machine Learning. 🤖
    """)

    # Gráfico de loss (erro) durante o treino
    st.subheader("📉 Curva de aprendizado (loss)")
    df_loss = pd.DataFrame({
        "Erro (MSE)": historico_treino.history["loss"]
    })
    st.line_chart(df_loss, use_container_width=True)
    st.caption("O erro cai conforme o modelo ajusta seus parâmetros a cada época.")


# ─── RODAPÉ ───────────────────────────────────────────────
st.divider()
st.caption(
    f"⚡ Baseado em ar-condicionado de {POTENCIA_WATTS}W · "
    f"Tarifa: R$ {tarifa_usuario:.2f}/kWh · "
    f"Modelo: TensorFlow/Keras Dense(1) · "
    f"Dados: simulados com ruído gaussiano"
)