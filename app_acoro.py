import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import datetime

# --- 1. CONFIGURACIÓN Y ESTILO PREMIUM ---
st.set_page_config(page_title="Centro de Inteligencia ACORO", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8faf9; }
    [data-testid="stSidebar"] { background-color: #0e231a !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; text-align: center;
    }
    h1 { color: #1e3d2f !important; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #2e5a47 0%, #1e3d2f 100%);
        color: white !important; border-radius: 10px; border: none; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CABECERA CENTRADA (LOGO Y TÍTULO) ---
with st.container():
    c1, c2, c3 = st.columns(3)
    with c2:
        try:
            st.image("logo_acoro.png", use_container_width=True)
        except:
            st.warning("⚠️ Logo 'logo_acoro.png' no encontrado")
        
    st.markdown("""
        <div style="text-align: center;">
            <h1>CENTRO DE INTELIGENCIA ACORO</h1>
            <p style='color: #555; font-size: 20px;'>Plataforma de Análisis de Datos en Medio Rural v2.0</p>
            <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, transparent, #2e5a47, transparent); margin-bottom: 30px;">
        </div>
    """, unsafe_allow_html=True)

# --- 3. BARRA DE BÚSQUEDA CENTRADA ---
col_b1, col_b2, col_b3 = st.columns(3)
with col_b2:
    busqueda = st.text_input("🔍 Filtrado Inteligente", placeholder="Escribe para buscar impacto (ej: agua, cosecha)...")

st.write("---")

# --- 4. CARGA DE DATOS E IA ---
try:
    memoria = np.load("memoria_ia.npz")
    vocab = memoria['vocabulario']; pesos = memoria['pesos']; bias = memoria['bias']
    st.sidebar.success(f"✅ IA Core: Online ({len(vocab)} palabras)")
except:
    st.sidebar.error("❌ IA Core: Offline. Ejecuta entrenar_ia.py")

with st.expander("📂 Panel de Entrada de Datos", expanded=True):
    archivo = st.file_uploader("Subir archivo de noticias (.txt)", type=["txt"])

if archivo:
    contenido = archivo.read().decode("utf-8")
    lineas = [l.strip() for l in contenido.split("\n") if l.strip()]
    
    resultados = []
    for frase in lineas:
        p_f = frase.lower().split()
        X = np.array([1 if p in p_f else 0 for p in vocab])
        prob = 1 / (1 + np.exp(-(np.dot(X, pesos) + bias)))
        sent = "Positivo" if prob > 0.5 else "Negativo"
        resultados.append({"Noticia": frase, "Análisis": sent, "Confianza": prob})
    
    df_total = pd.DataFrame(resultados)
    
    # Aplicar el filtro de la barra de búsqueda
    if busqueda:
        df = df_total[df_total['Noticia'].str.contains(busqueda, case=False)]
    else:
        df = df_total

    # --- 5. DASHBOARD DE RESULTADOS ---
    m1, m2, m3 = st.columns(3)
    pos = len(df[df['Análisis'] == "Positivo"])
    neg = len(df[df['Análisis'] == "Negativo"])
    
    with m1:
        st.markdown(f'<div class="metric-card"><h3>Total</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>Positivos</h3><h2 style="color:green;">{pos}</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>Negativos</h3><h2 style="color:red;">{neg}</h2></div>', unsafe_allow_html=True)

    st.write("###")
    st.dataframe(df.style.background_gradient(subset=['Confianza'], cmap='Greens'), use_container_width=True)

    # --- 6. INFORME PDF ---
    if st.button("📄 GENERAR INFORME ESTRATÉGICO"):
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "INFORME DE INTELIGENCIA ACORO", ln=True, align='C')
        pdf.output("Informe_Acoro.pdf")
        st.balloons()
        st.success("✅ Informe guardado como 'Informe_Acoro.pdf'")