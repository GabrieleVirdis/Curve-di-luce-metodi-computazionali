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

    parser = argparse.ArgumentParser(description='Plot delle curve di luce e spettri di potenza',
                                     usage      ='python3 Curve_di_luce.py  --opzione')
    
    # Settimanali
    parser.add_argument('--clw', action='store_true', help='Plot delle curve di luce settimanali')
    parser.add_argument('--spw', action='store_true', help='Plot dello spettro di potenza')
    parser.add_argument('--pfw', action='store_true', help='Fit dello spettro di potenza')
    parser.add_argument('--psw', action='store_true', help='Plot delle curve di luce sintentiche')
   
    # Mensili
    parser.add_argument('--clm', action='store_true', help='Plot delle curve di luce')
    parser.add_argument('--spm', action='store_true', help='Plot dello spettro di potenza')
    parser.add_argument('--spfm', action='store_true', help='Fit dello spettro di potenza')
    parser.add_argument('--psm', action='store_true', help='Plot delle curve di luce sintentiche')

    return parser.parse_args()
    
    

# ANALISI DELLE CURVE SETTIMANALI -----------

# Dizionario delle sorgenti
def main():
    
    args = parse_arguments()

    dcw_source = {
        '4FGL_J1104.4+3812': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_weekly_2_20_2025.csv',
        '4FGL_J1256.1-0547': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_weekly_2_20_2025.csv',
        '4FGL_J1555.7+1111': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_weekly_2_20_2025.csv',
        '4FGL_J2253.9+1609': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_weekly_2_20_2025.csv',
    }


    '''
    mese:

    # Dizionario delle sorgenti mensili
dcm_source = {
    '4FGL_J1104.4+3812': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1104.4+3812_monthly_2_20_2025.csv',
    '4FGL_J1256.1-0547': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1256.1-0547_monthly_2_20_2025.csv',
    '4FGL_J1555.7+1111': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J1555.7+1111_monthly_2_20_2025.csv',
    '4FGL_J2253.9+1609': 'https://raw.githubusercontent.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali/main/Dati/4FGL_J2253.9+1609_monthly_2_20_2025.csv',
}
    '''












    # Definizioni colonne
    flux = 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)'
    flux_err = 'Photon Flux Error(photons cm-2 s-1)'
    date = 'Julian Date'

    # Riempimento del dizionario con i dataframe
    dcfw_source = {}
    for source in dcw_source: 
        dfw_source = pd.read_csv(dcw_source[source])
        dfw_source[flux] = dfw_source[flux].str.replace('<', '')
        dfw_source[flux] = dfw_source[flux].astype(float)
        dfw_source[flux_err] = np.where(dfw_source[flux_err] == '-', 0, dfw_source[flux_err])
        dfw_source[flux_err] = dfw_source[flux_err].astype(float)
        dcfw_source[source] = dfw_source

    # Array di colori per i plot delle curve e dei fit    
    colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange']
    fit_colors = ['lime', 'red', 'cyan', 'gold']

    # Grafici delle curve di luce (--cdlw)
    if args.clw:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0
        for source in dcfw_source:
            axs[i].errorbar(dcfw_source[source][date], dcfw_source[source][flux], 
                          yerr=dcfw_source[source][flux_err], color=colors[i], label=source)
            axs[i].set_xlabel('Julian Date', fontsize=11)
            axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)', fontsize=10)
            axs[i].legend(loc='best')
            i += 1
        plt.suptitle('Grafico del flusso - Dati settimanali')
        plt.tight_layout()
        plt.show()

    # Calcolo della fft (necessario per sdpw e sdpfw)
    if args.spw or args.pfw:
        for source in dcfw_source:
            dt_w = dcfw_source[source][date][1] - dcfw_source[source][date][0]
            fft_w = fft.fft(dcfw_source[source][flux].values)
            freq_w = fft.fftfreq(len(fft_w), d=dt_w)
            dcfw_source[source]['fft_w'] = fft_w
            dcfw_source[source]['freq_w'] = freq_w

    # Plot dello spettro di potenza (scala log) (--sdpw)
    if args.spw:
        # Spettri separati (4 pannelli)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0
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
        plt.suptitle('Spettro di potenza - Dati settimanali')
        plt.tight_layout()
        plt.show()

        # Confronto spettri (singolo grafico)
        plt.subplots(figsize=(11, 7))
        i = 0
        for source in dcfw_source:
            plt.plot(dcfw_source[source]['freq_w'][:len(dcfw_source[source]['freq_w']) // 2], 
                    np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['freq_w']) // 2])**2, 
                    color=colors[i], label=source)
            i += 1
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('f [Hz]', fontsize=11)
        plt.ylabel(r'$|c_k|^2$', fontsize=11)
        plt.legend(fontsize=9, loc='best')
        plt.tick_params(labelsize=9)
        plt.title('Spettri di potenza settimanali - Confronto')
        plt.tight_layout()
        plt.show()

    # Fit degli spettri di potenza (--sdpfw) 
    if args.pfw:
        fit_params = {}
        for source in dcfw_source:
            freq = dcfw_source[source]['freq_w'][2:len(dcfw_source[source]['fft_w']) // 2]
            psw = np.absolute(dcfw_source[source]['fft_w'][2:len(dcfw_source[source]['fft_w']) // 2])**2
            pv, pc = optimize.curve_fit(noisef, freq, psw, p0=[1, 1])
            fit_params[source] = {'pv': pv, 'pc': pc}
            print(f'{source}: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')

        # Grafico fit degli spettri di potenza
        fig, axs = plt.subplots(2, 2, figsize=(15, 11))
        axs = axs.flatten()
        i = 0
        for source in dcfw_source:
            freq = dcfw_source[source]['freq_w'][:len(dcfw_source[source]['fft_w']) // 2]
            psd = np.absolute(dcfw_source[source]['fft_w'][:len(dcfw_source[source]['fft_w']) // 2])**2
            pv = fit_params[source]['pv']
            pc = fit_params[source]['pc']
            
            axs[i].plot(freq, psd, color=colors[i], label='Dati')
            axs[i].plot(freq[1:], noisef(freq[1:], pv[0], pv[1]), 
                       color=fit_colors[i], 
                       label=f'Fit: β = {pv[1]:.2f} ± {np.sqrt(pc[1,1]):.2f}')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel('f [1/days]', fontsize=11)
            axs[i].set_ylabel(r'$|c_k|^2$', fontsize=11)
            axs[i].set_title(source)
            axs[i].legend(fontsize=13, loc='best')
            axs[i].grid(True)
            i += 1
        plt.suptitle('Spettri di potenza con fit - Dati settimanali')
        plt.tight_layout()
        plt.show()


    # --- CURVE DI LUCE SINTETICHE (--clsw) ---
    if args.psw:
        np.random.seed(1728)
        df_rand_date = {}
        for source in dcfw_source:
            dfr_w = dcfw_source[source].copy()
            np.random.shuffle(dfr_w[date].values)
            df_rand_date[source] = dfr_w

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0
        for source in dcfw_source:
            axs[i].errorbar(df_rand_date[source][date], dcfw_source[source][flux], 
                          yerr=dcfw_source[source][flux_err], color=colors[i], label=source)
            axs[i].set_xlabel('Julian Date')
            axs[i].set_ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
            axs[i].legend(fontsize=12, loc='best')
            i += 1
        plt.suptitle('Grafico del flusso randomizzato - Dati settimanali')
        plt.tight_layout()
        plt.show()






















if __name__ == "__main__":

    main()
