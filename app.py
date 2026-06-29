import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from engine import FinancialEngine

# Configurazione del layout della pagina
st.set_page_config(page_title="Piattaforma Analisi AI", layout="wide", initial_sidebar_state="expanded")

# Stile CSS per il look scuro ultra-professionale
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.2rem !important; font-weight: 700; color: #f8fafc; }
    h3 { font-size: 1.4rem !important; font-weight: 600; color: #94a3b8; margin-top: 1.5rem; }
    .hot-card { background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 6px; margin-bottom: 10px; }
    .day-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }
    .metric-box { background-color: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

engine = FinancialEngine()

# --- BARRA LATERALE (SIDEBAR) PER SELEZIONARE L'ANALISI ---
st.sidebar.title("📊 Navigazione")
st.sidebar.markdown("Seleziona il tipo di analisi che desideri consultare:")

tipo_analisi = st.sidebar.radio(
    "Moduli Disponibili:",
    ["📈 Dashboard Predittiva Settimanale", "🔄 Modello Opzioni, Stop-Loss & Rischio"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Sviluppato con FinGPT & TradingAgents")


# --- SEZIONE 1: DASHBOARD SETTIMANALE ---
if tipo_analisi == "📈 Dashboard Predittiva Settimanale":
    st.title("📈 Financial Insights Dashboard")
    st.markdown("<p style='color: #64748b;'>Analisi predittiva multi-agente integrata</p>", unsafe_allow_html=True)
    
    st.write("### 🔥 AI Hot Scanner (Anteprima Titoli Caldi)")
    hot_list = engine.get_trending_stocks()
    cols_hot = st.columns(len(hot_list))
    for i, stock_info in enumerate(hot_list):
        with cols_hot[i]:
            st.markdown(f"""
                <div class="hot-card">
                    <span style="font-size: 1.2rem; font-weight: bold; color: #f8fafc;">{stock_info['ticker']}</span><br>
                    <span style="font-size: 0.85rem; color: #3b82f6; font-weight: 500;">{stock_info['reason']}</span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)

    st.write("### 🔍 Analisi Predittiva Settimanale")
    ticker_input = st.text_input("Inserisci un Ticker manualmente:", "NVDA", key="ticker_set").upper()

    if ticker_input:
        with st.spinner("Elaborazione modelli AI in corso..."):
            data = engine.analyze_ticker(ticker_input)
            
        st.markdown(f"#### Target: <span style='color: #3b82f6;'>{data['name']}</span> ({data['ticker']})", unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric(label="FinGPT Sentiment Score", value=f"{data['sentiment']}%")
        m2.metric(label="Previsione Statistica", value=data['prediction'])
        
        st.write("#### 📅 Strategia Operativa nei Giorni della Settimana")
        
        giorni_cols = st.columns(5)
        for idx, (giorno, dettagli) in enumerate(data['weekly'].items()):
            with giorni_cols[idx]:
                if "Acquisto" in dettagli['stato']:
                    color = "#10b981"
                elif "Vendita" in dettagli['stato']:
                    color = "#ef4444"
                else:
                    color = "#f59e0b"
                    
                st.markdown(f"""
                    <div class="day-card">
                        <b style="color: #94a3b8; font-size: 1.05rem;">{giorno}</b><br>
                        <p style="color: {color}; font-weight: bold; margin: 10px 0 5px 0; font-size: 0.95rem;">{dettagli['stato']}</p>
                        <span style="color: #64748b; font-size: 0.8rem;">Affidabilità: {dettagli['confidenza']}</span>
                    </div>
                """, unsafe_allow_html=True)


# --- SEZIONE 2: MODELLO OPZIONI E RISCHIO ---
elif tipo_analisi == "🔄 Modello Opzioni, Stop-Loss & Rischio":
    st.title("📊 Modello Predittivo Avanzato: Opzioni, Stop-Loss e Take-Profit")
    st.markdown("<p style='color: #64748b;'>Analisi statistica predittiva con Ensemble Learning e proiezione lineare dei prezzi a 7 giorni.</p>", unsafe_allow_html=True)
    
    ticker_input = st.text_input("Inserisci Ticker (es. PLTR, MU, ASML, NVDA):", "MU", key="ticker_opz").upper()
    
    if ticker_input:
        with st.spinner("Estrazione dati storici ed elaborazione indicatori avanzati..."):
            try:
                # 1. Download e pulizia dati (3 anni per garantire stabilità al Machine Learning)
                df = yf.Ticker(ticker_input).history(period="3y", auto_adjust=True)
                
                if not df.empty and len(df) > 50:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    # Calcolo RSI (14 periodi)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / (loss + 1e-10)
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Calcolo Bande di Bollinger (20 periodi, 2 deviazioni standard)
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['STD20'] = df['Close'].rolling(window=20).std()
                    df['Bollinger_Upper'] = df['MA20'] + (2 * df['STD20'])
                    df['Bollinger_Lower'] = df['MA20'] - (2 * df['STD20'])
                    
                    df['Distanza_Banda_Sup'] = (df['Close'] - df['Bollinger_Upper']) / df['Bollinger_Upper']
                    df['Distanza_Banda_Inf'] = (df['Close'] - df['Bollinger_Lower']) / df['Bollinger_Lower']
                    
                    # Calcolo Volumi Standardizzati (Z-Score)
                    df['Vol_Media20'] = df['Volume'].rolling(window=20).mean()
                    df['Vol_STD20'] = df['Volume'].rolling(window=20).std()
                    df['Volumi_Standardizzati'] = (df['Volume'] - df['Vol_Media20']) / (df['Vol_STD20'] + 1e-10)
                    
                    # Target Classificazione Opzioni (Orizzonte 7 giorni)
                    rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
                    
                    # SOLUZIONE INFALLIBILE: Sostituzione di np.select con logica nativa Python
                    target_classes = []
                    for val in rendimento_futuro_7g:
                        if pd.isna(val):
                            target_classes.append(0)
                        elif val >= 0.04:
                            target_classes.append(2)
                        elif val <= -0.03:
                            target_classes.append(1)
                        else:
                            target_classes.append(0)
                    df['Target_Class'] = target_classes
                    
                    # Target Regressione Prezzi Giornalieri (t+1 a t+5)
                    for i in range(1, 6):
                        df[f'Target_Price_t+{i}'] = df['Close'].shift(-i)
                    
                    # Estrazione ultimo stato reale prima di ripulire i NaN per il training
                    ultimo_stato = df.dropna(subset=['RSI', 'Bollinger_Upper', 'Volumi_Standardizzati']).iloc[-1].copy()
                    df_train = df.dropna().copy()
                    
                    # Caricamento delle Feature
                    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
                    X = df_train[features]
                    y_class = df_train['Target_Class']
                    
                    # 2. Train Ensemble Classificazione (Walk-Forward Validation)
                    split = int(len(df_train) * 0.8)
                    modello_rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                    modello_rf2 = RandomForestClassifier(n_estimators=150, min_samples_leaf=4, random_state=2026)
                    
                    modello_rf1.fit(X.iloc[:split], y_class.iloc[:split])
                    modello_rf2.fit(X.iloc[:split], y_class.iloc[:split])
                    accuratezza = modello_rf1.score(X.iloc[split:], y_class.iloc[split:])
                    
                    # Predizione Scenari
                    vettore_input = np.array([ultimo_stato[features].values])
                    prob1 = modello_rf1.predict_proba(vettore_input)[0]
                    prob2 = modello_rf2.predict_proba(vettore_input)[0]
                    probabilita_array = (prob1 + prob2) / 2
                    
                    classi_modello = list(modello_rf1.classes_)
                    prob_laterale = probabilita_array[classi_modello.index(0)] if 0 in classi_modello else 0.0
                    prob_stop_loss = probabilita_array[classi_modello.index(1)] if 1 in classi_modello else 0.0
                    prob_take_profit = probabilita_array[classi_modello.index(2)] if 2 in classi_modello else 0.0
                    
                    # 3. Train Modello di Regressione Prezzi
                    previsioni_prezzo = []
                    for i in range(1, 6):
                        y_reg = df_train[f'Target_Price_t+{i}']
                        modello_reg = LinearRegression()
                        modello_reg.fit(X, y_reg)
                        previsioni_prezzo.append(modello_reg.predict(vettore_input)[0])
                    
                    # --- INTERFACCIA GRAFICA ---
                    prezzo_attuale = ultimo_stato['Close']
                    st.write(f"### Analisi Predittiva del Rischio per {ticker_input} (Orizzonte 7 Giorni)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📈 Indicatori di Mercato Calcolati")
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Prezzo Attuale:</b> ${prezzo_attuale:.2f}<br>
                                <b>RSI (14 periodi):</b> <span style="color:#3b82f6; font-weight:bold;">{ultimo_stato['RSI']:.2f}</span><br>
                                <b>Banda BB Superiore:</b> ${ultimo_stato['Bollinger_Upper']:.2f}<br>
                                <b>Banda BB Inferiore (Supporto):</b> ${ultimo_stato['Bollinger_Lower']:.2f}<br>
                                <b>Z-Score Volumi:</b> {ultimo_stato['Volumi_Standardizzati']:.2f}
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("#### 🎯 Distribuzione delle Probabilità (Ensemble Model)")
                        st.markdown(f"""
                            <div class="metric-box">
                                <b>Accuratezza Backtest:</b> {accuratezza:.2%}<br>
                                <span style="color:#10b981; font-weight:bold;">🚀 Take Profit (≥ +4%):</span> {prob_take_profit:.1%}<br>
                                <span style="color:#3b82f6; font-weight:bold;">↔️ Scenario Laterale:</span> {prob_laterale:.1%}<br>
                                <span style="color:#ef4444; font-weight:bold;">⚠️ Stop Loss (≤ -3%):</span> {prob_stop_loss:.1%}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # --- 4. PREVISIONE PREZZI GIORNALIERI ---
                    st.write("---")
                    st.subheader("📅 Previsione del Prezzo Target nei Prossimi Giorni")
                    st.write("Prezzi puntuali attesi calcolati tramite regressione adattiva basata sulle metriche correnti.")
                    
                    # Generazione date escludendo weekend
                    giorni_settimana = []
                    data_corrente = datetime.now()
                    passo = 1
                    while len(giorni_settimana) < 5:
                        giorno_futuro = data_corrente + timedelta(days=passo)
                        if giorno_futuro.weekday() < 5:
                            giorni_settimana.append(giorno_futuro.strftime('%A (%d/%m)'))
                        passo += 1
                    
                    df_previsioni = pd.DataFrame({
                        'Giorno Previsto': giorni_settimana,
                        'Prezzo Target': [f"${p:.2f}" for p in previsioni_prezzo],
                        'Variazione Attesa': [f"{((p - prezzo_attuale) / prezzo_attuale):+.2%}" for p in previsioni_prezzo]
                    })
                    
                    c_tab, c_graf = st.columns([1, 1])
                    with c_tab:
                        st.dataframe(df_previsioni, use_container_width=True, hide_index=True)
                    with c_graf:
                        df_chart = pd.DataFrame({'Prezzo Target': previsioni_prezzo}, index=giorni_settimana)
                        st.line_chart(df_chart)
                        
                    st.success("Analisi statistica e forecast completati con successo.")
                else:
                    st.error("Dati insufficienti su Yahoo Finance per elaborare gli indicatori di questo ticker.")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione dell'analisi tecnica: {str(e)}")
