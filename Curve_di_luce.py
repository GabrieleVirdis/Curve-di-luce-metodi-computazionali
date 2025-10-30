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

### Dizionario sorgenti ###

sources = {
    '4FGL_J1104.4+3812': {
        'w': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_weekly_2_20_2025.csv',
        'm': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_monthly_2_20_2025.csv'
    },
    '4FGL_J1256.1-0547': {
        'w': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_weekly_2_20_2025.csv',
        'm': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_monthly_2_20_2025.csv'
    },
    '4FGL_J1555.7+1111': {
        'w': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_weekly_2_20_2025.csv',
        'm': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_monthly_2_20_2025.csv'
    },
    '4FGL_J2253.9+1609': {
        'w': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_weekly_2_20_2025.csv',
        'm': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_monthly_2_20_2025.csv'
    }
}

### Lettura e pulizia dati ###

data = {} # Creazione di un dataframe vuoto 

# Nuovi nomi delle colonne del dataframe
col_flux = 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)'
col_err = 'Photon Flux Error(photons cm-2 s-1)'
col_date = 'Julian Date'

# Riempimento del dataframe vuoto con i file presi dagli url del dizionario
for src in sources:
    data[src] = {}
    for t in sources[src]:
        df = pd.read_csv(sources[src][t])
        
        # Sostituzione dei - nella colonna degli errori del flusso con valori nulli
        mask = df[col_err].astype(str) == '-'
        df.loc[mask, col_err] = 0
        df[col_err] = df[col_err].astype(float)
        
        data[src][t] = df

### Grafici ###

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
fit_colors = ['#c0392b', '#2980b9', '#27ae60', '#e67e22']

# --- DATI SETTIMANALI ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs = axs.flatten()

for i, src in enumerate(data):
    d = data[src]['w']
    axs[i].errorbar(d[col_date], d[col_flux], yerr=d[col_err], 
                     capsize=4, color=colors[i], fmt= 'o',  markersize=4,
                    elinewidth=1.5, alpha=0.7, label=src)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Grafico del flusso - Dati settimanali', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()


'''
# --- DATI MENSILI ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten() # Trasforma la matrice 2x2 in un array a 1 dimensione

for i, src in enumerate(data):
    d = data[src]['m']
    axs[i].errorbar(d[col_date], d[col_flux], yerr=d[col_err],
                    fmt='o', capsize=4, color=colors[i], markersize=5,
                    elinewidth=1.5, alpha=0.7, label=src)
    axs[i].set_xlabel('Julian Date', fontsize=11)
    axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].grid(True, alpha=0.3, linestyle='--')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Grafico del flusso - Dati mensili', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()
'''

# --- ANALISI FOURIER ---
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
    
    psd = np.absolute(data[src]['freq_w'][:n])**2
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



'''
# Mensili

plt.subplots(figsize= (11, 7))

for i, src in enumerate(data):
    n = len(data[src]['freq_m']) // 2 
    psd = np.absolute(data[src]['fft_m'][:n])**2    
    plt.plot(data[src]['freq_m'][:n], psd, color=colors[i], linewidth=2, label=src)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('f [Hz]', fontsize=11)
plt.ylabel(r'$|c_k|^2$', fontsize=11)
plt.legend(fontsize=9, loc='best')
plt.tick_params(labelsize=9)
plt.title('Spettri di potenza mensili - Confronto', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()
'''



'''
# --- SPETTRO POTENZA MENSILE ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, src in enumerate(data):
    n = len(data[src]['freq_m']) // 2

    psd = np.absolute(data[src]['fft_m'][:n])**2
    axs[i].plot(freq_m[:n], psd, color=colors[i], linewidth=2, label=src)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel('f [Hz]', fontsize=11)
    axs[i].set_ylabel(r'$|c_k|^2$', fontsize=11)
    axs[i].legend(fontsize=9, loc='best')
    axs[i].tick_params(labelsize=9)

plt.suptitle('Spettro di potenza - Dati mensili', fontsize=14, y=0.995)
plt.tight_layout()
plt.show()
'''

# Calcolo fit per tutte le sorgenti
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


