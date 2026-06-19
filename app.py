import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURAZIONE PAGINA STREAMLIT ---
st.set_page_config(page_title="QuantModel Opzioni & Risk", layout="wide")
st.title("📊 Modello Predittivo Avanzato: Opzioni, Stop-Loss & Take-Profit")
st.write("Analisi statistica predittiva a 7 giorni con RSI, Bande di Bollinger e Volumi Standardizzati.")

# --- 1. DOWNLOAD E FEATURE ENGINEERING AVANZATA ---
@st.cache_data
def calcola_modello_avanzato(ticker_symbol):
    try:
        # Scarichiamo 3 anni di dati giornalieri
        df = yf.Ticker(ticker_symbol).history(period="3y")
        if df.empty:
            return None
        
        # --- A. CALCOLO RSI (14 periodi) ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # --- B. BANDE DI BOLLINGER (20 periodi, 2 deviazioni standard) ---
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['MA20'] + (2 * df['STD20'])
        df['Bollinger_Lower'] = df['MA20'] - (2 * df['STD20'])
        # Indicatori di posizione rispetto alle bande (normalizzati)
        df['Distanza_Banda_Sup'] = (df['Close'] - df['Bollinger_Upper']) / df['Bollinger_Upper']
        df['Distanza_Banda_Inf'] = (df['Close'] - df['Bollinger_Lower']) / df['Bollinger_Lower']
        
        # --- C. VOLUMI STANDARDIZZATI (Z-Score a 20 giorni) ---
        # Indica se i volumi odierni sono insolitamente alti rispetto alla media recente
        df['Vol_Media20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_STD20'] = df['Volume'].rolling(window=20).std()
        df['Volumi_Standardizzati'] = (df['Volume'] - df['Vol_Media20']) / (df['Vol_STD20'] + 1e-10)
        
        # --- D. DEFINIZIONE TARGET MULTI-CLASS (Orizzonte 7 giorni) ---
        # Rendimento percentuale nei successivi 7 giorni di borsa
        rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
        
        # Logica delle condizioni: 2 se Target >= +4%, 1 se Target <= -3%, 0 altrimenti
        condizioni = [
            (rendimento_futuro_7g >= 0.04),
            (rendimento_futuro_7g <= -0.03)
        ]
        scelte = [2, 1] # 2 = Take Profit, 1 = Stop Loss
        df['Target'] = np.select(condizioni, scelte, default=0)
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

# --- 2. INTERFACCIA UTENTE (SIDEBAR) ---
st.sidebar.header("Parametri Modello")
ticker = st.sidebar.text_input("Inserisci Ticker (es. PLTR, MU, ASML, NVDA):", value="PLTR").upper()

df_mercato = calcola_modello_avanzato(ticker)

if df_mercato is None or df_mercato.empty:
    st.warning("Inserisci un ticker valido nella barra laterale per avviare il calcolo.")
else:
    # Definizione delle feature per il Machine Learning
    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
    
    X = df_mercato[features]
    y = df_mercato['Target']
    
    # Split temporale (Walk-forward)
    split = int(len(df_mercato) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Addestriamo il classificatore
    modello = RandomForestClassifier(n_estimators=150, random_state=42)
    modello.fit(X_train, y_train)
    accuratezza = modello.score(X_test, y_test)
    
    # --- 3. DASHBOARD VISIVA ---
    st.subheader(f"Analisi Predittiva del Rischio per {ticker} (Orizzonte 7 Giorni)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📈 Indicatori di Mercato Calcolati")
        ultimo_stato = df_mercato[features].iloc[-1]
        
        st.metric(label="RSI (14 periodi)", value=f"{ultimo_stato['RSI']:.2f}")
        st.write(f"**Posizione vs Banda Bollinger Superiore:** {ultimo_stato['Distanza_Banda_Sup']:.2%}")
        st.write(f"**Posizione vs Banda Bollinger Inferiore:** {ultimo_stato['Distanza_Banda_Inf']:.2%}")
        st.metric(label="Z-Score Volumi (Standardizzati)", value=f"{ultimo_stato['Volumi_Standardizzati']:.2f}")
        
        st.write("---")
        calcola = st.button("Calcola Probabilità di Scenario")
        
    with col2:
        st.markdown("### 🎯 Probabilità Statistiche per Opzioni & Livelli")
        st.metric(label="Accuratezza Storica del Modello", value=f"{accuratezza:.2%}")
        
        if calcola:
            vettore_input = np.array([ultimo_stato.values])
            # predict_proba restituisce le probabilità per ciascuna classe [Classe 0, Classe 1, Classe 2]
            probabilita = modello.predict_proba(vettore_input)[0]
            
            prob_laterale = probabilita[0] # Classe 0
            prob_stop_loss = probabilita[1] # Classe 1 (-3%)
            prob_take_profit = probabilita[2] # Classe 2 (+4%)
            
            st.write("---")
            st.markdown("#### Distribuzione delle Probabilità nei prossimi 7 giorni:")
            
            # Visualizzazione dinamica basata sul valore più alto
            col_tp, col_sideways, col_sl = st.columns(3)
            with col_tp:
                st.success(f"🚀 **Take Profit (+4%):** \n\n **{prob_take_profit:.1%}**")
            with col_sideways:
                st.info(f"↔️ **Laterale (-3% / +4%):** \n\n **{prob_laterale:.1%}**")
            with col_sl:
                st.warning(f"⚠️ **Stop Loss (-3%):** \n\n **{prob_stop_loss:.1%}**")
            
            # Grafico a barre orizzontale
            chart_data = pd.DataFrame({
                'Scenario': ['Stop Loss (≤ -3%)', 'Laterale (Canale)', 'Take Profit (≥ +4%)'],
                'Probabilità': [prob_stop_loss, prob_laterale, prob_take_profit]
            })
            st.bar_chart(data=chart_data, x='Scenario', y='Probabilità', use_container_width=True)
            
            st.markdown("#### Prezzo Storico Ultimi 100 Giorni")
            st.line_chart(df_mercato['Close'].tail(100))
        else:
            st.info("Clicca sul pulsante per elaborare gli indicatori reali ed estrarre le probabilità matematiche.")