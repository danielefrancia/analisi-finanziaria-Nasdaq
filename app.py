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
        # Forziamo il ticker pulito ed in maiuscolo
        ticker_clean = str(ticker_symbol).strip().upper()
        
        # Scarichiamo 3 anni di dati giornalieri
        df = yf.Ticker(ticker_clean).history(period="3y")
        if df.empty:
            return None, None
        
        # Ripuliamo l'eventuale multi-index introdotto dalle nuove versioni di yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
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
        df['Vol_Media20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_STD20'] = df['Volume'].rolling(window=20).std()
        df['Volumi_Standardizzati'] = (df['Volume'] - df['Vol_Media20']) / (df['Vol_STD20'] + 1e-10)
        
        # --- D. DEFINIZIONE TARGET MULTI-CLASS (Orizzonte 7 giorni) ---
        rendimento_futuro_7g = (df['Close'].shift(-7) - df['Close']) / df['Close']
        
        condizioni = [
            (rendimento_futuro_7g >= 0.04),
            (rendimento_futuro_7g <= -0.03)
        ]
        scelte = [2, 1] # 2 = Take Profit, 1 = Stop Loss
        df['Target'] = np.select(condizioni, choices=scelte, default=0)
        
        # SALVIAMO L'ULTIMA RIGA ASSOLUTA (Oggi) PRIMA DEL DROPNA
        # Altrimenti i dati degli ultimi 7 giorni verrebbero cancellati perché non hanno ancora il target futuro!
        ultimo_disponibile = df.dropna(subset=['RSI', 'Bollinger_Upper', 'Volumi_Standardizzati']).iloc[-1].copy()
        
        # df_train conterrà solo i dati passati con target valido per addestrare il Machine Learning
        df_train = df.dropna().copy()
        
        return df_train, ultimo_disponibile
    except Exception as e:
        st.error(f"Errore nel calcolo: {e}")
        return None, None

# --- 2. INTERFACCIA UTENTE (SIDEBAR) ---
st.sidebar.header("Parametri Modello")
ticker_input = st.sidebar.text_input("Inserisci Ticker (es. PLTR, MU, ASML, NVDA):", value="PLTR")

df_mercato, ultimo_stato = calcola_modello_avanzato(ticker_input)

if df_mercato is None or df_mercato.empty or ultimo_stato is None:
    st.warning("Inserisci un ticker valido nella barra laterale per avviare il calcolo.")
else:
    # Definizione delle feature per il Machine Learning
    features = ['RSI', 'Distanza_Banda_Sup', 'Distanza_Banda_Inf', 'Volumi_Standardizzati']
    ticker = ticker_input.upper()
    
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
        
        # Prezzo Attuale Reale estratto in sicurezza
        prezzo_attuale = ultimo_stato['Close']
        banda_inf = ultimo_stato['Bollinger_Lower']
        
        st.metric(label="Prezzo Attuale", value=f"${prezzo_attuale:.2f}")
        st.metric(label="RSI (14 periodi)", value=f"{ultimo_stato['RSI']:.2f}")
        st.write(f"**Posizione vs Banda Bollinger Superiore:** {ultimo_stato['Distanza_Banda_Sup']:.2%}")
        st.write(f"**Posizione vs Banda Bollinger Inferiore:** {ultimo_stato['Distanza_Banda_Inf']:.2%}")
        st.write(f"**Banda Inferiore (Supporto):** ${banda_inf:.2f}")
        st.metric(label="Z-Score Volumi (Standardizzati)", value=f"{ultimo_stato['Volumi_Standardizzati']:.2f}")
        
        st.write("---")
        calcola = st.button("Calcola Probabilità di Scenario")
        
    with col2:
        st.markdown("### 🎯 Probabilità Statistiche per Opzioni & Livelli")
        st.metric(label="Accuratezza Storica del Modello", value=f"{accuratezza:.2%}")
        
        # Calcolo dinamico dei target basati sul prezzo attuale reale
        target_tp = prezzo_attuale * 1.04
        target_sl = prezzo_attuale * 0.97
        
        col_metrics_1, col_metrics_2 = st.columns(2)
        with col_metrics_1:
            st.metric(label="Take-Profit Consigliato (Target +4%)", value=f"${target_tp:.2f}")
        with col_metrics_2:
            st.metric(label="Stop-Loss Rigido (Protezione -3%)", value=f"${target_sl:.2f}")
            
        if calcola:
            # Estraiamo solo i valori delle feature nell'ordine corretto
            vettore_input = np.array([ultimo_stato[features].values])
            probabilita = modello.predict_proba(vettore_input)[0]
            
            prob_laterale = probabilita[0]  # Classe 0
            prob_stop_loss = probabilita[1]  # Classe 1 (-3%)
            prob_take_profit = probabilita[2] # Classe 2 (+4%)
            
            st.write("---")
            st.markdown("#### Distribuzione delle Probabilità nei prossimi 7 giorni:")
            
            col_tp, col_sideways, col_sl = st.columns(3)
            with col_tp:
                st.success(f"🚀 **Take Profit (≥ +4%):** \n\n **{prob_take_profit:.1%}**")
            with col_sideways:
                st.info(f"↔️ **Laterale (Canale):** \n\n **{prob_laterale:.1%}**")
            with col_sl:
                st.warning(f"⚠️ **Stop Loss (≤ -3%):** \n\n **{prob_stop_loss:.1%}**")
            
            # Grafico a barre orizzontale
            chart_data = pd.DataFrame({
                'Scenario': ['Stop Loss (≤ -3%)', 'Laterale (Canale)', 'Take Profit (≥ +4%)'],
                'Probabilità': [prob_stop_loss, prob_laterale, prob_take_profit]
            })
            st.bar_chart(data=chart_data, x='Scenario', y='Probabilità', use_container_width=True)
            
            st.markdown("#### Prezzo Storico Ultimi 100 Giorni")
            # Usiamo df_mercato per mostrare lo storico completo senza buchi
            st.line_chart(df_mercato['Close'].tail(100))
        else:
            st.info("Clicca sul pulsante per elaborare gli indicatori reali ed estrarre le probabilità matematiche.")
