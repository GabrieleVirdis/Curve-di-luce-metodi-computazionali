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

### --- FUNZIONE PER IL FIT SPETTRO DI POTENZA ---

def noisef(f, N, beta):
    """
    Funzione per fit Spettro Potenza di diversi tipi di rumore

    f    : frequenze
    N    : normalizzazione
    beta : esponente per dipendenza da frequenza

    return N/f^beta
    """

    return N/f**beta


### ANALISI SETTIMANALE ###

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

    dfw_source[flux] = pd.to_numeric(dfw_source[flux].astype('string').str.replace('<', ''))
    dfw_source[flux_err] = pd.to_numeric(dfw_source[flux_err].replace('-', '0'))
    
    dcfw_source[source] = dfw_source # riempimento del nuovo dizionario


# Pulizia dei limit sup ed errori incerti


### Grafici ###

colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange']
fit_colors = ['lime', 'red', 'cyan', 'gold']  # Aggiunti colori per il fit


# --- GRAFICI SORGENTI ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs = axs.flatten()

for i, source in enumerate(dcfw_source): # enumerate perché così fai il counter solo con le chiavi del dizionario che sono le sorgenti 
    axs[i].errorbar(dcfw_source[source][date], dcfw_source[source][flux], yerr=dcfw_source[source][flux_err], 
                     capsize=4, color=colors[i], fmt='o', markersize=4,
                    elinewidth=1.5, alpha=0.7, label=source)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Grafico del flusso - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


# >>> ANALISI DI FOURIER
for source in dcfw_source:
  
    dt_w = dcfw_source[source][date][1] - dcfw_source[source][date][0] # Intervallo di campionamento in giorni tra due misure consecutive
    fft_w = fft.fft(np.array(dcfw_source[source][flux].values , dtype=float)) # Calcolo dei coefficenti di Fourier
    freq_w = fft.fftfreq(len(fft_w), d=dt_w) # Calcolo delle frequenze
    
    # Salva FFT (rimosso fft_m e freq_m che non sono definiti)
    dcfw_source[source]['fft_w'] = fft_w
    dcfw_source[source]['freq_w'] = freq_w


# --- SPETTRO POTENZA SETTIMANALE ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, source in enumerate(dcfw_source):

    axs[i].plot(dcfw_source[source]['freq_w'][:len(dcfw_source[source]['fft_w']) // 2], 
                np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['fft_w']) // 2])**2, 
                color=colors[i], linewidth=2, label=source)
    
    axs[i].set_xscale('log') 
    axs[i].set_yscale('log')
    axs[i].set_xlabel('f [Hz]', fontsize=11) 
    axs[i].set_ylabel(r'$|c_k|^2$', fontsize=11)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Spettro di potenza - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


# --- CONFRONTO TRA GLI SPETTRI DI POTENZA --- 

# Settimanali
plt.subplots(figsize=(11, 7))

for i, source in enumerate(dcfw_source):
    
    plt.plot(dcfw_source[source]['freq_w'][:len(dcfw_source[source]['freq_w']) // 2], 
            np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['freq_w']) // 2])**2, 
            color=colors[i], linewidth=2, label=source) 

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
for i, source in enumerate(dcfw_source):
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

plt.suptitle('Spettri di potenza con fit - Dati settimanali', fontsize=15, y=0.998)
plt.tight_layout()
plt.show()


# Inizializzazione del seed 
np.random.seed(1728)

# Randomizza unicamente le misure temporali
df_rand_date = {}
for source in dcfw_source:
    # Dati settimanali
    df_w = dcfw_source[source].copy()
    np.random.shuffle(df_w[date].values)

    df_rand_date[source] = {'w': df_w}

# --- GRAFICI DATI SETTIMANALI RANDOMIZZATI---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
for i, source in enumerate(df_rand_date):
    d = df_rand_date[source]['w']
    axs[i].errorbar(d[date], d[flux], yerr=d[flux_err],
                     capsize=4, color=colors[i], fmt='o', markersize=4,
                    elinewidth=1.5, alpha=0.7, label=source)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)
plt.suptitle('Grafico del flusso Randomizzato - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

