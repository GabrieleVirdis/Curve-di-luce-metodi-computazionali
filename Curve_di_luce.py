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
    
    dfw_source = pd.read_csv ( dcw_source[source] ) # lettura delle sorgenti.csv

    dfw_source[flux] = pd.to_numeric(dfw_source[flux].astype('string').str.replace('<', ''))
    dfw_source[flux_err] = pd.to_numeric(dfw_source[flux_err].replace('-', '0'))
    
    dcfw_source[source] = dfw_source # riempimento del nuovo dizionario


# Pulizia dei limit sup ed errori incerti


### Grafici ###

colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange']


# --- GRAFICI SORGENTI ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs = axs.flatten()

for i, source in enumerate(dcfw_source): # enumerate perché così fai il counter solo con le chiavi del dizionario che sono le sorgenti 
    axs[i].errorbar(dcfw_source[source][date], dcfw_source[source][flux], yerr=dcfw_source[source][flux_err], 
                     capsize=4, color=colors[i], fmt= 'o',  markersize=4,
                    elinewidth=1.5, alpha=0.7, label=source)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Grafico del flusso - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


'''
for src in data:
    # Settimanale
    dt_w = data[src]['w'][col_date][1] - data[src]['w'][col_date][0] # Intervallo di campionamento in giorni tra due misure consecutive
    fft_w = fft.fft(data[src]['w'][col_flux].values)
    freq_w = fft.fftfreq(len(fft_w), d=dt_w)
    
    # Mensile
    dt_m = data[src]['m'][col_date][1] - data[src]['m'][col_date][0]
    fft_m = fft.fft(data[src]['m'][col_flux].values)
    freq_m = fft.fftfreq(len(fft_m), d=dt_m)
    
    # Salva FFT
    data[src].update({'fft_w' : fft_w,  'freq_w' : freq_w, 'fft_m' : fft_m,  'freq_m' : freq_m})


# --- SPETTRO POTENZA SETTIMANALE ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, src in enumerate(data):

    n = len(data[src]['fft_w']) // 2
    
    psd = np.absolute(data[src]['fft_w'][:n])**2
    axs[i].plot(data[src]['freq_w'][:n], psd, color=colors[i], linewidth=2, label=src)
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

# Settimali
plt.subplots(figsize= (11, 7))

for i, src in enumerate(data):
    n = len(data[src]['freq_w']) // 2 
    psd = np.absolute(data[src]['fft_w'][:n])**2    
    plt.plot(data[src]['freq_w'][:n], psd, color=colors[i], linewidth=2, label=src)

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

for src in data:
    n = len(data[src]['fft_w']) // 2
    freq = data[src]['freq_w'][2:n]
    psd = np.absolute(data[src]['fft_w'][2:n])**2
    
    pv, pc = optimize.curve_fit(noisef, freq, psd, p0=[1, 1])
    fit_params[src] = {'pv': pv, 'pc': pc}
    print(f'{src}: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')

# Grafico con 4 pannelli (uno per ogni sorgente)
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
axs = axs.flatten()

# Un pannello per ogni sorgente
for i, src in enumerate(data):
    n = len(data[src]['fft_w']) // 2
    freq = data[src]['freq_w'][:n]
    psd = np.absolute(data[src]['fft_w'][:n])**2
    
    pv = fit_params[src]['pv']
    pc = fit_params[src]['pc']
    
    # Dati
    axs[i].plot(freq, psd, color=colors[i], linewidth=2, alpha=0.7, label='Dati')
    
    # Fit
    axs[i].plot(freq[1:], noisef(freq[1:], pv[0], pv[1]), 
                color=fit_colors[i], linewidth=2.5, linestyle='--', 
                label=f'Fit: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')
    
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel('f [1/days]', fontsize=11)
    axs[i].set_ylabel(r'$|c_k|^2', fontsize=11)
    axs[i].set_title(src, fontsize=12, fontweight='bold')
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)
    axs[i].grid(True, alpha=0.3, linestyle=':')

plt.suptitle('Spettri di potenza con fit - Dati settimanali', fontsize=15,  y=0.998)
plt.tight_layout()
plt.show()


# Inizializzaizione del seed 
np.random.seed(1728)

# Randomizza unicamente le misure temporali
df_rand_date = {}
for src in data:
    # Dati settimanali
    df_w = data[src]['w'].copy()
    np.random.shuffle(df_w[col_date].values)

    df_rand_date[src] = {'w': df_w, 'm': df_m}

# --- GRAFICI DATI SETTIMANALI RANDOMIZZATI---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
for i, src in enumerate(df_rand_date):
    d = df_rand_date[src]['w']
    axs[i].errorbar(d[col_date], d[col_flux], yerr=d[col_err],
                     capsize=4, color=colors[i], fmt='o', markersize=4,
                    elinewidth=1.5, alpha=0.7, label=src)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)
plt.suptitle('Grafico del flusso Randomizzato - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()
'''
