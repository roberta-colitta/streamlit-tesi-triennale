# streamlit app to create an inteeractive demonstration of the algorithm

from datetime import datetime, timedelta
from subprocess import call
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from cox_ross_rubernstein import CRR
from customized_pascal import *
from black_scholes import BlackScholes
from monte_carlo import *


def _add(d1, d2):
    d1.update(d2)
    return d1


st.title("Sviluppo computazionale dei principali modelli di pricing delle opzioni​")
st.text(
    "​Collezioni di applicazioni interattive per la dimostrazione della tesi finale. \nRoberta Colitta"
)
st.text("Seleziona una modalità dall'interfaccia laterale per il tema appropriato.")

with st.sidebar:
    st.title("Selezione modalità")
    mode = st.selectbox(
        "Selezione modalità", ["Cox-Ross-Rubenstein", "Black Scholes", "Monte Carlo"]
    )

    if mode == "Cox-Ross-Rubenstein":
        st.subheader("Seleziona la configurazione per Put/Call")
        D = st.date_input("Seleziona data", value=datetime.today())
        E = st.date_input(
            "Data di scadenza", value=datetime.today() + timedelta(days=23)
        )

        T = (E - D).total_seconds() / (60 * 60 * 24 * 365)
        S = st.number_input(
            "Prezzo del titolo sottostante (S)",
            value=24533.84,
            min_value=0.0,
            max_value=10000000.0,
        )
        K = st.number_input(
            "Strike Price (K)", value=25000.0, min_value=0.0, max_value=10000000.0
        )
        rf = 0.01 * st.number_input(
            "Tasso Risk-free (rf)", value=1.315, min_value=0.000, max_value=100.000
        )
        sigma = st.number_input(
            "Volatilità (σ)", value=0.243, min_value=0.000, max_value=1.000
        )
        depth = st.slider(
            label="Numero di step dell'albero binomiale",
            min_value=2,
            max_value=10,
            value=3,
        )

    elif mode == "Black Scholes":
        st.subheader("Seleziona la configurazione per Put/Call")
        D = st.date_input("Seleziona data", value=datetime.today())
        E = st.date_input(
            "Data di scadenza", value=datetime.today() + timedelta(days=74)
        )

        S_select = st.number_input(
            "Prezzo del titolo sottostante (S)",
            value=25002.74,
            min_value=0.0,
            max_value=10000000.0,
        )

        K = st.number_input(
            "Strike Price (K)", value=24000.0, min_value=0.0, max_value=10000000.0
        )

        S_low = S_select * 0.8

        S_hi = S_select * 1.2

        S = np.arange(S_low, S_hi, 100)

        rf = 0.01 * st.number_input(
            "Tasso Risk-free (rf)", value=1.107, min_value=0.000, max_value=100.000
        )
        sigma = st.number_input(
            "Volatilità (σ)", value=0.244, min_value=0.0, max_value=1.0
        )
        T = (E - D).total_seconds() / (60 * 60 * 24 * 365)

    elif mode == "Monte Carlo":
        st.subheader("Seleziona la configurazione per Put/Call")
        D = st.date_input("Seleziona data", value=datetime.today())
        E = st.date_input(
            "Data di scadenza", value=datetime.today() + timedelta(days=139)
        )
        S_select = st.number_input(
            "Prezzo del titolo sottostante (S)",
            value=24252.16,
            min_value=0.0,
            max_value=10000000.0,
        )

        S_low = S_select * 0.8

        S_hi = S_select * 1.2

        S = np.arange(S_low, S_hi, 100)
        
        # find the index in S where S_select is the closest value
        S_index = np.argmin(np.abs(S - S_select))

        K = st.number_input(
            "Strike Price (K)", value=27000.0, min_value=0.0, max_value=10000000.0
        )
        rf = 0.01 * st.number_input(
            "Tasso Risk-free (rf)", value=1.107, min_value=0.000, max_value=100.000
        )
        sigma = st.number_input("Volatilità", value=0.202, min_value=0.0, max_value=1.0)
        T = (E - D).total_seconds() / (60 * 60 * 24 * 365)


if mode == "Cox-Ross-Rubenstein":
    st.header("Simulazione del modello Cox-Ross-Rubenstein")
    st.text("Configurazione applicata:")

    c1 = CRR(S=S, K=K, rf=rf, sigma=sigma, N=depth, tipo_opzione="Call")
    c2 = CRR(S=S, K=K, rf=rf, sigma=sigma, N=depth, tipo_opzione="Put")

    data = {
        "Simbolo": ["S", "K", "T (convertito in anni)", "rf", "σ"],
        "Opzione Call": [S, K, T, rf, sigma],
        "Opzione Put": [S, K, T, rf, sigma],
    }
    input_frame = pd.DataFrame(
        data,
        columns=["Simbolo", "Opzione Call", "Opzione Put"],
        index=[
            "Prezzo del titolo sottostante",
            "Strike price",
            "Tempo alla maturità",
            "Tasso risk-free",
            "Volatilità",
        ],
    )

    st.dataframe(input_frame)
    tree_with_probabilities = c1.get_tree_with_probs(T=T)
    call_tree = c1.build_tree_option(T=T)
    put_tree = c2.build_tree_option(T=T)

    st.subheader(
        "Valore delle opzioni europee Call e Put con il modello di Cox, Ross e Rubinstein"
    )
    out_frame = pd.DataFrame(
        {
            "Tipo Opzione": ["Call", "Put"],
            "Prezzo": [call_tree[-1][0], put_tree[-1][0]],
        }
    )

    st.dataframe(out_frame)

    tp = [
        f"{int(item[0])} \n p={round(item[1], 2)}"
        for sublist in tree_with_probabilities
        for item in sublist
    ]

    fig, ax = GraphTriangle(len(tree_with_probabilities)).plot_digraph(custom_labels=tp)
    st.subheader("Andamento dei prezzi del titolo Sottostante S")
    st.pyplot(fig)
    tp = [f"{int(item)}" for sublist in call_tree for item in sublist]

    fig, ax = GraphTriangle(len(call_tree)).plot_digraph(custom_labels=tp[::-1])
    st.subheader("Andamento dei prezzi dell'opzione Call")
    st.pyplot(fig)

    tp = [f"{int(item)}" for sublist in put_tree for item in sublist]

    fig, ax = GraphTriangle(len(call_tree)).plot_digraph(custom_labels=tp[::-1])
    st.subheader("Andamento dei prezzi dell'opzione Put")
    st.pyplot(fig)

if mode == "Black Scholes":
    st.header("Simulazione del modello Black Scholes")
    st.text("Configurazione applicata:")
    data = {
        "Simbolo": ["S", "K", "T (convertito in anni)", "rf", "σ"],
        "Opzione Call": [S_select, K, T, rf, sigma],
        "Opzione Put": [S_select, K, T, rf, sigma],
    }
    input_frame = pd.DataFrame(
        data,
        columns=["Simbolo", "Opzione Call", "Opzione Put"],
        index=[
            "Prezzo del titolo sottostante",
            "Strike price",
            "Tempo alla maturità",
            "Tasso risk-free",
            "Volatilità",
        ],
    )

    tipo = st.selectbox(
        "Tipo di Modello",
        ["modello classico", "con il dividend yield", "con i costi di transazione"],
    )
    kwargs = {}

    if tipo == "con il dividend yield":
        dy = 0.01 * st.number_input(
            "Dividend Yield (%)", value=4.180, min_value=0.000, max_value=100.000
        )
        kwargs["q"] = dy
    elif tipo == "con i costi di transazione":
        phi = 0.01 * st.number_input(
            "Phi (%)", value=1.956, min_value=0.000, max_value=100.000
        )
        interval = st.number_input(
            "Intervallo", value=0.25, min_value=0.000, max_value=50.000
        )

        kwargs["phi"] = phi
        kwargs["interval"] = interval

    st.dataframe(input_frame)

    calls = [
        BlackScholes(
            s, K=K, rf=rf, sigma=sigma, T=T, tipo_opzione="Call", **kwargs
        ).get_option_price()
        for s in S
    ]
    puts = [
        BlackScholes(
            s, K=K, rf=rf, sigma=sigma, T=T, tipo_opzione="Put", **kwargs
        ).get_option_price()
        for s in S
    ]

    call_select = BlackScholes(
        S=S_select,
        K=K,
        rf=rf,
        sigma=sigma,
        T=T,
        tipo_opzione="Call",
        **kwargs,
    ).get_option_price()

    put_select = BlackScholes(
        S=S_select,
        K=K,
        rf=rf,
        sigma=sigma,
        T=T,
        tipo_opzione="Put",
        **kwargs,
    ).get_option_price()
    st.subheader(
        "Valore delle opzioni europee Call e Put con il modello di Black Scholes"
    )
    out_frame = pd.DataFrame(
        {
            "Tipo Opzione": ["Call", "Put"],
            "Prezzo": [call_select, put_select],
        }
    )
    
    st.dataframe(out_frame)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(S, calls, label="Call")
    ax.plot(S, puts, label="Put")
    plt.xlabel("$S_0$")
    plt.ylabel(" Valore")
    ax.legend()

    st.pyplot(fig)

if mode == "Monte Carlo":
    st.header("Simulazione del modello Monte Carlo")
    # tipo = st.selectbox("Tipo opzione", ["Europea", "Asiatica"])
    tipo = "Europea"
    if tipo == "Europea":
        model = MC_Simulation(mc_european)
    elif tipo == "Asiatica":
        model = MC_Simulation(mc_asian)
    st.text("Configurazione applicata:")

    data = {
        "Simbolo": ["S", "K", "T (convertito in anni)", "rf", "sigma"],
        "Opzione Call": [S_select, K, T, rf, sigma],
        "Opzione Put": [S_select, K, T, rf, sigma],
    }
    input_frame = pd.DataFrame(
        data,
        columns=["Simbolo", "Opzione Call", "Opzione Put"],
        index=[
            "Prezzo del titolo sottostante",
            "Strike price",
            "tempo alla maturità",
            "Tasso risk-free",
            "Volatilità",
        ],
    )
    st.dataframe(input_frame)

    iteration = st.slider(
        "Numero di Iterazioni", min_value=1, max_value=10000, value=500
    )
    if iteration > 3000:
        st.warning("potrebbe richiedere un po di tempo per il calcolo")

    base_kwargs = {
        "k": K,
        "r": rf,
        "sigma": sigma,
        "T": T,
        "timesteps": 50,
    }

    puts = []
    bar = st.progress(0.0)
    for i, s in enumerate(S):
        puts.append(
            model.run(
                iteration,
                **_add(
                    base_kwargs,
                    {
                        "s": s,
                        "option_type": "Put",
                    },
                ),
            )
        )
        bar.progress(i / (len(S) * 2))

    calls = []
    for i, s in enumerate(S):
        calls.append(
            model.run(
                iteration,
                **_add(
                    base_kwargs,
                    {
                        "s": s,
                        "option_type": "Call",
                    },
                ),
            )
        )
        bar.progress(0.5 + i / (len(S) * 2))
        
    st.subheader(
        "Valore delle opzioni europee Call e Put con il modello di Monte Carlo"
    )
    out_frame = pd.DataFrame(
        {
            "Tipo Opzione": ["Call", "Put"],
            "Prezzo": [calls[S_index], puts[S_index]],
        }
    )

    st.dataframe(out_frame)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(S, calls, label="Call")
    ax.plot(S, puts, label="Put")
    plt.xlabel("$S_0$")
    plt.ylabel(" Valore")
    ax.legend()

    st.pyplot(fig)
