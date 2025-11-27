'''
Gabriele Virdis (gabriele.virdis@studenti.unipg.it)

- Università degli Studi di Perugia
- Corso di Metodi Computazionali per la Fisica
'''

import sys, os
import numpy as np
import pandas as pd
from scipy import constants, fft, optimize
import matplotlib.pyplot as plt


import argparse

# Funzione per il fit dello spettro di potenza

def noisef(f, N, beta):
    """
    Funzione per fit Spettro Potenza di diversi tipi di rumore

    f    : frequenze
    N    : normalizzazione
    beta : esponente per dipendenza da frequenza

    return N/f^beta
    """

    return N/f**beta


def parse_arguments():

    parser = argparse.ArgumentParser(description='plot delle curve di luce, spettro di potenza e curve di luce sintetiche',
                                     usage      ='python3 Curve_di_luce.py  --opzioni')
    parser.add_argument('--cdlw', action='store_true', help='Plot delle curve di luce settimanali')
    parser.add_argument('--sdpw', action='store_true', help='Plot dello spettro di potenza')
    parser.add_argument('--sdpfw', action='store_true', help='Fit dello spettro di potenza')
    parser.add_argument('--clsw', action='store_true', help='Plot delle curve di luce sintentiche')
   

    # Mensile
'''
    parser.add_argument('--cdlm', action='store_true', help='Plot delle curve di luce')
    parser.add_argument('--sdpm', action='store_true', help='Plot dello spettro di potenza')
    parser.add_argument('--sdpfm', action='store_true', help='Fit dello spettro di potenza')
    parser.add_argument('--clsm', action='store_true', help='Plot delle curve di luce sintentiche')

''' 



    return  parser.parse_args(args=None if sys.argv[1:] else ['--help'])


# ANALISI SETTIMANALE

# Dizionario delle sorgenti settimanali
dcw_source = {
    '4FGL_J1104.4+3812': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_weekly_2_20_2025.csv',
    '4FGL_J1256.1-0547': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_weekly_2_20_2025.csv',
    '4FGL_J1555.7+1111': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_weekly_2_20_2025.csv',
    '4FGL_J2253.9+1609': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_weekly_2_20_2025.csv',
}

# Dizionario delle sorgenti mensili
dcm_source = {
    '4FGL_J1104.4+3812': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_monthly_2_20_2025.csv',
    '4FGL_J1256.1-0547': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_monthly_2_20_2025.csv',
    '4FGL_J1555.7+1111': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_monthly_2_20_2025.csv',
    '4FGL_J2253.9+1609': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_monthly_2_20_2025.csv',
}

# Nuovi nomi delle colonne del dataframe
flux = 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)'
flux_err = 'Photon Flux Error(photons cm-2 s-1)'
date = 'Julian Date'

# Creo un nuovo dizionario contenente i dataframe settimanali

dcfw_source = { } 

# Ciclo per aggiungere al nuovo dizionario i dataframe
for source in dcw_source: 
    
    dfw_source = pd.read_csv(dcw_source[source]) # lettura delle sorgenti.csv
    
    dfw_source[flux] = dfw_source[flux].str.replace('<' , '') # Permette di vedere tutta la stringa e usare il valore limite superiore
    dfw_source[flux] = dfw_source[flux].astype(float) # riporto tutto float sennò sono tipo stringe o oggetto

    dfw_source[flux_err] = np.where(dfw_source[flux_err] == '-', 0 , dfw_source[flux_err]) # Uso where al posto di replace sennò errore dovuto alla notazione scientifica
    dfw_source[flux_err] = dfw_source[flux_err].astype(float)

    dcfw_source[source] = dfw_source # riempimento del nuovo dizionario

### Grafici ###

colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange'] # Colori per i grafici per suddividere le sorgenti
fit_colors = ['lime', 'red', 'cyan', 'gold']  # Insieme di colori per il fit


# --- GRAFICI SORGENTI ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs = axs.flatten() # converte in array 1D per fare il ciclo for

i=0 # Inizializzo i
for source in dcfw_source:
        axs[i].errorbar(dcfw_source[source][date], dcfw_source[source][flux], yerr=dcfw_source[source][flux_err], color=colors[i], label=source)
        axs[i].set_xlabel('Julian Date', fontsize=11)
        axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
        axs[i].legend(fontsize=9, loc='best')
        i += 1 # aumento il contatore

plt.suptitle('Grafico del flusso - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


# ANALISI DI FOURIER
for source in dcfw_source:
  
    dt_w = dcfw_source[source][date][1] - dcfw_source[source][date][0] # Intervallo di campionamento in giorni tra due misure consecutive
    fft_w = fft.fft(dcfw_source[source][flux].values) # Calcolo dei coefficenti di Fourier
    freq_w = fft.fftfreq(len(fft_w), d=dt_w) # Calcolo delle frequenze
    
    # Salva FFT (rimosso fft_m e freq_m che non sono definiti)
    dcfw_source[source]['fft_w'] = fft_w
    dcfw_source[source]['freq_w'] = freq_w


# --- SPETTRO POTENZA SETTIMANALE ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()


i= 0 
for source in dcfw_source:
    axs[i].plot(dcfw_source[source]['freq_w'][:len(dcfw_source[source]['fft_w']) // 2], 
                np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['fft_w']) // 2])**2, 
                color=colors[i], linewidth=2, label=source)
    
    axs[i].set_xscale('log') 
    axs[i].set_yscale('log')
    axs[i].set_xlabel('f [Hz]', fontsize=11) 
    axs[i].set_ylabel(r'$|c_k|^2$', fontsize=11)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)
    i += 1 

plt.suptitle('Spettro di potenza - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


# --- CONFRONTO TRA GLI SPETTRI DI POTENZA --- 

# Settimanali
plt.subplots(figsize=(11, 7))

i = 0
for source in dcfw_source:
    
    plt.plot(dcfw_source[source]['freq_w'][:len(dcfw_source[source]['freq_w']) // 2], 
            np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['freq_w']) // 2])**2, 
            color=colors[i], linewidth=2, label=source) 
    i += 1
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('f [Hz]', fontsize=11)
plt.ylabel(r'$|c_k|^2$', fontsize=11)
plt.legend(fontsize=9, loc='best')
plt.tick_params(labelsize=9)
plt.title('Spettri di potenza settimanali - Confronto', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

# ---CALCOLO DEL FIT PER LE TUTTE LE SORGENTI SETTIMANALI---
fit_params = {}

for source in dcfw_source:
    freq = dcfw_source[source]['freq_w'][2:len(dcfw_source[source]['fft_w']) // 2] 
    psw = np.absolute(dcfw_source[source]['fft_w'][2:len(dcfw_source[source]['fft_w']) // 2])**2
    
    pv, pc = optimize.curve_fit(noisef, freq, psw, p0=[1, 1])
    fit_params[source] = {'pv': pv, 'pc': pc}
    print(f'{source}: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')

# Grafico con 4 pannelli (uno per ogni sorgente)
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
axs = axs.flatten()

# Un pannello per ogni sorgente
i = 0
for source in dcfw_source:


    freq = dcfw_source[source]['freq_w'][:len(dcfw_source[source]['fft_w']) // 2]
    psd = np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['fft_w']) // 2])**2
    
    pv = fit_params[source]['pv']
    pc = fit_params[source]['pc']
    
    # Dati
    axs[i].plot(freq, psd, color=colors[i], linewidth=2, alpha=0.7, label='Dati')
    
    # Fit
    axs[i].plot(freq[1:], noisef(freq[1:], pv[0], pv[1]), 
                color=fit_colors[i], linewidth=2.5, linestyle='--', 
                label=f'Fit: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')
    
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel('f [1/days]', fontsize=11)
    axs[i].set_ylabel(r'$|c_k|^2$', fontsize=11)
    axs[i].set_title(source, fontsize=12)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)
    axs[i].grid(True, alpha=0.3, linestyle=':')

    i += 1



plt.suptitle('Spettri di potenza con fit - Dati settimanali', fontsize=15, y=0.998)
plt.tight_layout()
plt.show()


# Inizializzazione del seed 
np.random.seed(1728)

# Randomizza unicamente le misure temporali
df_rand_date = {}
for source in dcfw_source:
    # Dati settimanali
    dfr_w = dcfw_source[source].copy()
    np.random.shuffle(dfr_w[date].values)

    df_rand_date[source] = dfr_w

# --- GRAFICI DATI SETTIMANALI RANDOMIZZATI---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

i=0 # Inizializzo i
for source in dcfw_source:
    axs[i].errorbar(df_rand_date[source][date], dcfw_source[source][flux], yerr=dcfw_source[source][flux_err], color=colors[i], label=source)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    i += 1 # aumento il contatore

plt.suptitle('Grafico del flusso randomizzato - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

















