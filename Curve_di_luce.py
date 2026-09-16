'''
Gabriele Virdis (gabriele.virdis@studenti.unipg.it)

- Università degli Studi di Perugia
- Corso di Metodi Computazionali per la Fisica
'''

# Import librerie
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft, optimize

# Definizione delle funzioni
def fit_potenza(frequencies, N, beta):
    """
    Funzione che descrive lo spettro di potenza in funzione della frequenza

    Parametri
    -----------
        frequencies: frequenze dello spettro
        N:           fattore di normalizzazione
        beta:        esponente della dipendenza dalla frequenza

    Restituisce
    -----------
        N/frequencies**beta
    """
    return N/frequencies**beta

def trova_picco_spettro(frequencies, power):
    """
    Funzione che trova il picco massimo dello spettro di potenza e il periodo corrispondente

    Parametri
    -----------
        frequencies: frequenze dello spettro
        power:       potenze associate alle frequenze (|C_k|^2)

    Restituisce
    -----------
        frequenza, potenza e periodo corrispondenti al picco massimo
    """

    max_power = np.max(power) # Trova il valore massimo nell'array delle potenze
    max_frequencies = frequencies[power == max_power][0] # Applica una maschera all'array delle frequenze per trovare la frequenza corrispondente al picco massimo e poi estrae il primo valore     
    max_period = 1 / max_frequencies

    return max_frequencies, max_power, max_period


def genera_curve_sintetiche(source_data, flux, flux_err, number_synthetic_curves):
    """
    Funzione che genera curve di luce sintetiche per ogni sorgente riordinando casualmente flussi ed errori

    Parametri
    -----------
        source_data:             dizionario contenente i dati delle sorgenti
        flux:                    nome della colonna del flusso
        flux_err:                nome della colonna dell'errore del flusso
        number_synthetic_curves: numero di curve sintetiche da generare

    Restituisce
    -----------
        Un dizionario contenente le curve sintetiche per ogni sorgente, dove le chiavi sono i nomi delle sorgenti e i valori sono liste di DataFrame contenenti le curve sintetiche
    """

    np.random.seed(1717) # Seed per generare numeri casuali
    synthetic_curves = {} # Dizionario vuoto per contenere le curve sintetiche per ogni sorgente

    for source in source_data: # Ciclo su tutte le sorgenti 
        synthetic_curves[source] = [] # Crea una lista vuota per contenere le curve sintetiche 
        source_df = source_data[source]['df'] 

        for i in range(number_synthetic_curves): # Ciclo per ripetere il rimescolamento per il numero di curve sintetiche 
            synthetic_curve = source_df.copy() # Copia del dataframe per creare una curva sintetica

            # Rimescolamento
            order = np.arange(len(synthetic_curve)) # Array che contiene gli indici della curva di luce
            np.random.shuffle(order) # Rimescola casualmente gli indici 
            synthetic_curve[flux] = source_df[flux].values[order] # Rimescola i valori del flusso in base agli indici rimescolati prima
            synthetic_curve[flux_err] = source_df[flux_err].values[order] # Rimescola i valori dell'errore del flusso in base agli indici rimescolati prima

            synthetic_curves[source].append(synthetic_curve) # Aggiunge la curva sintetica alla lista iniziale

    return synthetic_curves


def parse_arguments():
    parser = argparse.ArgumentParser(description='Grafici delle curve di luce e degli spettri di potenza',
                                     usage='python3 Curve_di_luce.py --option')

    # Settimanali
    parser.add_argument('--clw',    action='store_true',  help='Mostra le curve di luce settimanali')
    parser.add_argument('--spw',    action='store_true',  help='Mostra e analizza gli spettri di potenza settimanali')
    parser.add_argument('--pfw',    action='store_true',  help='Esegue il fit degli spettri di potenza settimanali')
    parser.add_argument('--psw',    action='store_true',  help='Mostra e analizza gli spettri di potenza sintetici settimanali') 

    # Mensili
    parser.add_argument('--clm',    action='store_true',   help='Mostra le curve di luce mensili')
    parser.add_argument('--spm',    action='store_true',   help='Mostra e analizza gli spettri di potenza mensili')
    parser.add_argument('--pfm',    action='store_true',   help='Esegue il fit degli spettri di potenza mensili' )
    parser.add_argument('--psm',    action='store_true',   help='Mostra e analizza gli spettri di potenza sintetici mensili')

    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])


def main():
  
    args = parse_arguments()

    # Numero di curve sintetiche generate
    number_synthetic_curves = 1000

    # Definizione delle colonne
    flux = 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)'
    flux_err = 'Photon Flux Error(photons cm-2 s-1)'
    date = 'Julian Date'

    # Definizione dei colori dei grafici e dei fit
    colors = ['darkgreen', 'darkred', 'darkblue', 'darkorange']
    fit_colors = ['lime', 'red', 'cyan', 'gold']

# ANALISI SETTIMANALE ------------------------------------------

    # Dizionario sorgenti settimanali con i percorsi dei file csv
    weekly_source_files = {
        '4FGL_J1104.4+3812': 'Dati/4FGL_J1104.4+3812_weekly_2_20_2025.csv',
        '4FGL_J1256.1-0547': 'Dati/4FGL_J1256.1-0547_weekly_2_20_2025.csv',
        '4FGL_J1555.7+1111': 'Dati/4FGL_J1555.7+1111_weekly_2_20_2025.csv',
        '4FGL_J2253.9+1609': 'Dati/4FGL_J2253.9+1609_weekly_2_20_2025.csv',
    }

    # Trasformazione dei dati settimanali in dataFrame
    weekly_source_data = {} # Dizionario vuoto per contenere i dati delle sorgenti settimanali

    for source in weekly_source_files: # Ciclo su tutte le sorgenti per leggere i file csv e creare i dataFrame
        weekly_df = pd.read_csv(weekly_source_files[source])

        # Gestione dei limiti superiori
        upper_limits = weekly_df[flux_err] == '-' # Maschera per identificare i limiti superiori (dove l'errore è '-')
        for index in weekly_df.loc[upper_limits].index: # ciclo sugli indici delle righe con limiti superiori 
            weekly_df.loc[index, flux] = weekly_df.loc[index, flux][1:] # Rimuove il simbolo '<' dal flusso che è il primo carattere della stringa
            weekly_df.loc[index, flux_err] = 0 # Imposta l'errore nullo perchè non è definito per i limiti superiori

        # Conversione delle colonne di flusso e di errore in valori numerici
        weekly_df[flux] = weekly_df[flux].astype(float)
        weekly_df[flux_err] = weekly_df[flux_err].astype(float)

        weekly_source_data[source] = {'df': weekly_df} # Aggiunta dei dataframe al dizionario 

    # Grafici delle curve di luce settimanali
    if args.clw:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        for source in weekly_source_data:
            axs[i].errorbar(weekly_source_data[source]['df'][date], weekly_source_data[source]['df'][flux], yerr=weekly_source_data[source]['df'][flux_err], 
                            color=colors[i], label=source)
            axs[i].set_xlabel('Data giuliana')
            axs[i].set_ylabel(r'Flusso [fotoni cm$^{-2}$ s$^{-1}$]')
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Curve di luce - Dati settimanali')
        plt.tight_layout()
        plt.show()

    # Calcolo dei coefficenti di fourier e frequenze 
    for source in weekly_source_data:
        source_df = weekly_source_data[source]['df']
        dt = source_df[date][1] - source_df[date][0] # Calcolo dell'intervallo di tempo tra due osservazioni
        c = fft.fft(source_df[flux].values) # Calcolo dei coefficienti di Fourier della curva di luce settimanale
        f = fft.fftfreq(len(c), d=dt) # Calcolo delle frequenze corrispondenti ai coefficienti di Fourier
        weekly_source_data[source].update({'c': c, 'freq': f}) # Aggiunta dei coefficenti di fourier e delle frequenze al dizionario

    # Grafici degli spettri di potenza settimanali 
    if args.spw:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        # Analisi dei picchi degli spettri di potenza settimanali
        for source in weekly_source_data:
            # La ricerca del picco esclude le prime due componenti di frequenza e le frequenze negative
            tmp_len = len(weekly_source_data[source]['c']) // 2
            frequencies_w = weekly_source_data[source]['freq'][2:tmp_len]
            powers_w = np.absolute(weekly_source_data[source]['c'][2:tmp_len]) ** 2
            max_frequency, max_power, max_period = trova_picco_spettro(frequencies_w, powers_w) # Chiamata della funzione per trovare il picco massimo

            # Calcolo del tempo di osservazione e del numero di cicli osservati durante il periodo di misurazione
            period_years = max_period / 365 # Conversione del periodo da giorni ad anni
            observation_time = (weekly_source_data[source]['df'][date].iloc[-1] - weekly_source_data[source]['df'][date].iloc[0]) # Tempo di misurazione totale
            number_periods = observation_time / max_period # Calcolo del numero di cicli osservati durante il periodo di misurazione

            # Print dei risultati 
            print('\nSorgente: {}\nFrequenza massima: {:.2e} 1/giorni\n' 'Potenza massima: {:.2e}\nPeriodo: {:.2f} giorni ({:.2f} anni)\n'
                  'Numero di periodi nel tempo di misurazione: {:.2f}'.format(source, max_frequency, max_power, max_period, period_years, number_periods))

            axs[i].plot(frequencies_w, powers_w, color=colors[i], label=source)
            axs[i].axvline(max_frequency, color='black', linestyle='--', label='Periodo: {:.2f} giorni'.format(max_period))
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Spettri di potenza - dati settimanali')
        plt.tight_layout()
        plt.show()

        # Grafico di confronto tra gli spettri delle quattro sorgenti
        plt.subplots(figsize=(11, 7))
        i = 0

        for source in weekly_source_data:
            tmp_len = len(weekly_source_data[source]['c']) // 2
            frequencies_w = weekly_source_data[source]['freq'][2:tmp_len]
            powers_w = np.absolute(weekly_source_data[source]['c'][2:tmp_len]) ** 2
            plt.plot(frequencies_w, powers_w, color=colors[i], label=source)
            i += 1

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Frequenza [giorni$^{-1}$]')
        plt.ylabel(r'$|c_k|^2$')
        plt.legend(loc='best')
        plt.title('Confronto spettri di potenza settimanali')
        plt.tight_layout()
        plt.show()

    # Grafici dei fit degli spettri settimanali
    if args.pfw:
        fig, axs = plt.subplots(2, 2, figsize=(15, 11))
        axs = axs.flatten()
        i = 0

        # Il fit usa lo stesso intervallo di frequenze scelto per i picchi
        for source in weekly_source_data:
            tmp_len = len(weekly_source_data[source]['c']) // 2
            f = weekly_source_data[source]['freq'][2:tmp_len]
            psw = np.absolute(weekly_source_data[source]['c'][2:tmp_len]) ** 2
            max_frequency, max_power, max_period = trova_picco_spettro(f, psw) # Chiamata della funzione per trovare il picco massimo
            pv, pc = optimize.curve_fit(fit_potenza, f, psw, p0=[1e-16, 1])

            # Print dei risultati del fit
            print('\nSorgente: {}\nN = {:.2e} ± {:.2e}\n' 'β = {:.2f} ± {:.2f}'.format(source, pv[0], np.sqrt(pc[0, 0]),
                    pv[1], np.sqrt(pc[1, 1])))

            axs[i].plot(f, psw, color=colors[i], label=source)
            axs[i].plot(f, fit_potenza(f, pv[0], pv[1]), color=fit_colors[i], label=f'Fit: β = {pv[1]:.2f} ± {np.sqrt(pc[1, 1]):.2f}')
            axs[i].axvline(max_frequency, color='black', linestyle='--', label='Periodo: {:.2f} giorni'.format(max_period))
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].legend(fontsize=13, loc='best')
            axs[i].grid(True)
            i += 1
        plt.suptitle('Fit degli spettri di potenza - dati settimanali')
        plt.tight_layout()
        plt.show()

    # Grafici degli spettri sintetici e significatività dei picchi settimanali
    if args.psw:
        synthetic_curves_w = genera_curve_sintetiche(weekly_source_data, flux, flux_err, number_synthetic_curves) # Chiamata della funzione per generare le curve sintetiche settimanali    

        # Calcolo degli spettri di potenza sintetici
        synthetic_spectra_w = {} # Dizionario vuoto per contenere gli spettri di potenza sintetici per ogni sorgente
        for source in synthetic_curves_w:
            synthetic_spectra_w[source] = [] # Crea una lista vuota per contenere gli spettri di potenza sintetici 
            for synthetic_curve in synthetic_curves_w[source]:
                dt = synthetic_curve[date][1] - synthetic_curve[date][0]
                c = fft.fft(synthetic_curve[flux].values)
                f = fft.fftfreq(len(c), d=dt)

                synthetic_spectra_w[source].append({'c': c, 'freq': f} ) # Aggiunta al dizionario 

        # Calcolo della potenza massima per ogni curva sintetica
        synthetic_max_power_w = {} # Dizionario vuoto per contenere le potenze massime sintetiche per ogni sorgente
        for source in synthetic_spectra_w:
            synthetic_max_power_w[source] = [] # Crea una lista vuota per contenere le potenze massime sintetiche 
            for spectrum in synthetic_spectra_w[source]: # Ciclo su tutti gli spettri di potenza sintetici
                tmp_len = len(spectrum['c']) // 2
                max_power = trova_picco_spettro(spectrum['freq'][2:tmp_len], np.absolute(spectrum['c'][2:tmp_len]) ** 2)[1] # Chiamata della funzione per trovare il picco massimo

                synthetic_max_power_w[source].append(max_power) # Aggiunta al dizionario 
            synthetic_max_power_w[source] = np.array(synthetic_max_power_w[source]) # Conversione della lista delle potenze massime sintetiche in un array

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        # Grafici log-log degli spettri di potenza sintetici settimanali
        for source in synthetic_spectra_w:
            number = 0

            for spectrum in synthetic_spectra_w[source]: # Ciclo su tutti gli spettri di potenza sintetici 
                tmp_len = len(spectrum['c']) // 2
                label = source if number == 0 else None # Solo alla prima curva per evitare duplicati
                axs[i].plot(spectrum['freq'][2:tmp_len], np.absolute(spectrum['c'][2:tmp_len]) ** 2, alpha=0.01, color=colors[i], label=label)
                number += 1

            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].set_title(f'{number_synthetic_curves} spettri sintetici')
            axs[i].legend(loc='best')
            i += 1
        plt.suptitle('Spettri di potenza sintetici - Dati settimanali')
        plt.tight_layout()
        plt.show()

        # Confronto tra le potenze massime sintetiche e quella reale
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        print('\nAnalisi degli spettri sintetici settimanali')

        for source in synthetic_spectra_w:
            tmp_len = len(weekly_source_data[source]['c']) // 2
            real_frequencies = weekly_source_data[source]['freq'][2:tmp_len]
            real_power = np.absolute(weekly_source_data[source]['c'][2:tmp_len]) ** 2
            real_max_power = trova_picco_spettro(real_frequencies, real_power)[1] # Chiamata della funzione per trovare il picco massimo

            # Confronto tra le potenze massime sintetiche e quella reale
            synthetic_max_power = synthetic_max_power_w[source] # Potenze massime sintetiche della sorgente
            number_exceeding = np.sum(synthetic_max_power >= real_max_power) # Conteggio delle potenze massime sintetiche che superano o eguagliano quella reale
            number_synthetic = len(synthetic_max_power) # Conteggio del numero totale di potenze massime sintetiche
            probability_percentage = number_exceeding / number_synthetic * 100 # Calcolo della probabilità di ottenere un picco almeno altrettanto rilevante in percentuale

            # Print dei risultati dell'analisi dei picchi sintetici
            print('\nSorgente: {}\nPotenza massima reale: {:.2e}\n' 'Curve sintetiche con un picco almeno altrettanto rilevante: {} su {}\n'
                  'Probabilità di ottenere un picco almeno altrettanto rilevante: {:.1f}%'.format(source, real_max_power, number_exceeding,
                    number_synthetic, probability_percentage))

            # Istogrammi che confrontano i massimi sintetici e della potenza massima reale
            axs[i].hist(synthetic_max_power, bins=5, color=colors[i], edgecolor='black', label='Massimi sintetici')
            axs[i].axvline(real_max_power, color='black', linestyle='--', label='Massimo reale')
            axs[i].set_xscale('log')
            axs[i].set_xlabel(r'Massima potenza $|c_k|^2$')
            axs[i].set_ylabel('Conteggio')
            axs[i].set_title(source)
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Distribuzione dei massimi sintetici - Dati settimanali')
        plt.tight_layout()
        plt.show()

# ANALISI MENSILE ------------------------------------------

    # Dizionario sorgenti mensili con i percorsi dei file csv
    monthly_source_files = {
        '4FGL_J1104.4+3812': 'Dati/4FGL_J1104.4+3812_monthly_2_20_2025.csv',
        '4FGL_J1256.1-0547': 'Dati/4FGL_J1256.1-0547_monthly_2_20_2025.csv',
        '4FGL_J1555.7+1111': 'Dati/4FGL_J1555.7+1111_monthly_2_20_2025.csv',
        '4FGL_J2253.9+1609': 'Dati/4FGL_J2253.9+1609_monthly_2_20_2025.csv',
    }

    # Trasformazione dei dati mensili in dataFrame
    monthly_source_data = {} # Dizionario vuoto per contenere i dati delle sorgenti mensili

    for source in monthly_source_files: # Ciclo su tutte le sorgenti per leggere i file csv e creare i dataFrame
        monthly_df = pd.read_csv(monthly_source_files[source])

        # Gestione dei limiti superiori
        upper_limits = monthly_df[flux_err] == '-' # Maschera per identificare i limiti superiori (dove l'errore è '-')
        for index in monthly_df.loc[upper_limits].index: # ciclo sugli indici delle righe con limiti superiori
            monthly_df.loc[index, flux] = monthly_df.loc[index, flux][1:] # Rimuove il simbolo '<' dal flusso che è il primo carattere della stringa
            monthly_df.loc[index, flux_err] = 0 # Imposta l'errore nullo perchè non è definito per i limiti superiori

        # Conversione delle colonne di flusso e di errore in valori numerici
        monthly_df[flux] = monthly_df[flux].astype(float)
        monthly_df[flux_err] = monthly_df[flux_err].astype(float)
        monthly_source_data[source] = {'df': monthly_df} # Aggiunta dei dataframe al dizionario

    # Grafici delle curve di luce mensili
    if args.clm:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        for source in monthly_source_data:
            axs[i].errorbar(monthly_source_data[source]['df'][date], monthly_source_data[source]['df'][flux], yerr=monthly_source_data[source]['df'][flux_err],
                            color=colors[i], label=source)
            axs[i].set_xlabel('Data giuliana')
            axs[i].set_ylabel(r'Flusso [fotoni cm$^{-2}$ s$^{-1}$]')
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Curve di luce - Dati mensili')
        plt.tight_layout()
        plt.show()

    # Calcolo dei coefficenti di fourier e frequenze
    for source in monthly_source_data:
        source_df = monthly_source_data[source]['df']
        dt = source_df[date][1] - source_df[date][0] # Calcolo dell'intervallo di tempo tra due osservazioni
        c = fft.fft(source_df[flux].values) # Calcolo dei coefficienti di Fourier della curva di luce mensile
        f = fft.fftfreq(len(c), d=dt) # Calcolo delle frequenze corrispondenti ai coefficienti di Fourier
        monthly_source_data[source].update({'c': c, 'freq': f}) # Aggiunta dei coefficenti di fourier e delle frequenze al dizionario

    # Grafici degli spettri di potenza mensili
    if args.spm:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        # Analisi dei picchi degli spettri di potenza mensili
        for source in monthly_source_data:
            # La ricerca del picco esclude le prime due componenti di frequenza e le frequenze negative
            tmp_len = len(monthly_source_data[source]['c']) // 2
            frequencies_m = monthly_source_data[source]['freq'][2:tmp_len]
            powers_m = np.absolute(monthly_source_data[source]['c'][2:tmp_len]) ** 2
            max_frequency, max_power, max_period = trova_picco_spettro(frequencies_m, powers_m) # Chiamata della funzione per trovare il picco massimo

            # Calcolo del tempo di osservazione e del numero di cicli osservati durante il periodo di misurazione
            period_years = max_period / 365 # Conversione del periodo da giorni ad anni
            observation_time = (monthly_source_data[source]['df'][date].iloc[-1] - monthly_source_data[source]['df'][date].iloc[0]) # Tempo di misurazione totale
            number_periods = observation_time / max_period # Calcolo del numero di cicli osservati durante il periodo di misurazione

            # Print dei risultati
            print('\nSorgente: {}\nFrequenza massima: {:.2e} 1/giorni\n' 'Potenza massima: {:.2e}\nPeriodo: {:.2f} giorni ({:.2f} anni)\n'
                  'Numero di periodi nel tempo di misurazione: {:.2f}'.format(source, max_frequency, max_power, max_period, period_years, number_periods))

            axs[i].plot(frequencies_m, powers_m, color=colors[i], label=source)
            axs[i].axvline(max_frequency, color='black', linestyle='--', label='Periodo: {:.2f} giorni'.format(max_period))
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Spettri di potenza - dati mensili')
        plt.tight_layout()
        plt.show()

        # Grafico di confronto tra gli spettri delle quattro sorgenti
        plt.subplots(figsize=(11, 7))
        i = 0

        for source in monthly_source_data:
            tmp_len = len(monthly_source_data[source]['c']) // 2
            frequencies_m = monthly_source_data[source]['freq'][2:tmp_len]
            powers_m = np.absolute(monthly_source_data[source]['c'][2:tmp_len]) ** 2
            plt.plot(frequencies_m, powers_m, color=colors[i], label=source)
            i += 1

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Frequenza [giorni$^{-1}$]')
        plt.ylabel(r'$|c_k|^2$')
        plt.legend(loc='best')
        plt.title('Confronto spettri di potenza mensili')
        plt.tight_layout()
        plt.show()

    # Grafici dei fit degli spettri mensili
    if args.pfm:
        fig, axs = plt.subplots(2, 2, figsize=(15, 11))
        axs = axs.flatten()
        i = 0

        # Il fit usa lo stesso intervallo di frequenze scelto per i picchi
        for source in monthly_source_data:
            tmp_len = len(monthly_source_data[source]['c']) // 2
            f = monthly_source_data[source]['freq'][2:tmp_len]
            psm = np.absolute(monthly_source_data[source]['c'][2:tmp_len]) ** 2
            max_frequency, max_power, max_period = trova_picco_spettro(f, psm) # Chiamata della funzione per trovare il picco massimo
            pv, pc = optimize.curve_fit(fit_potenza, f, psm, p0=[1e-16, 1])

            # Print dei risultati del fit
            print('\nSorgente: {}\nN = {:.2e} ± {:.2e}\n' 'β = {:.2f} ± {:.2f}'.format(source, pv[0], np.sqrt(pc[0, 0]),
                    pv[1], np.sqrt(pc[1, 1])))

            axs[i].plot(f, psm, color=colors[i], label=source)
            axs[i].plot(f, fit_potenza(f, pv[0], pv[1]), color=fit_colors[i], label=f'Fit: β = {pv[1]:.2f} ± {np.sqrt(pc[1, 1]):.2f}')
            axs[i].axvline(max_frequency, color='black', linestyle='--', label='Periodo: {:.2f} giorni'.format(max_period))
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].legend(fontsize=13, loc='best')
            axs[i].grid(True)
            i += 1
        plt.suptitle('Fit degli spettri di potenza - dati mensili')
        plt.tight_layout()
        plt.show()

    # Grafici degli spettri sintetici e significatività dei picchi mensili
    if args.psm:
        synthetic_curves_m = genera_curve_sintetiche(monthly_source_data, flux, flux_err, number_synthetic_curves) # Chiamata della funzione per generare le curve sintetiche mensili

        # Calcolo degli spettri di potenza sintetici
        synthetic_spectra_m = {} # Dizionario vuoto per contenere gli spettri di potenza sintetici per ogni sorgente
        for source in synthetic_curves_m:
            synthetic_spectra_m[source] = [] # Crea una lista vuota per contenere gli spettri di potenza sintetici
            for synthetic_curve in synthetic_curves_m[source]:
                dt = synthetic_curve[date][1] - synthetic_curve[date][0]
                c = fft.fft(synthetic_curve[flux].values)
                f = fft.fftfreq(len(c), d=dt)
                synthetic_spectra_m[source].append({'c': c, 'freq': f} ) # Aggiunta al dizionario

        # Calcolo della potenza massima per ogni curva sintetica
        synthetic_max_power_m = {} # Dizionario vuoto per contenere le potenze massime sintetiche per ogni sorgente
        for source in synthetic_spectra_m:
            synthetic_max_power_m[source] = [] # Crea una lista vuota per contenere le potenze massime sintetiche
            for spectrum in synthetic_spectra_m[source]: # Ciclo su tutti gli spettri di potenza sintetici
                tmp_len = len(spectrum['c']) // 2
                max_power = trova_picco_spettro(spectrum['freq'][2:tmp_len], np.absolute(spectrum['c'][2:tmp_len]) ** 2)[1] # Chiamata della funzione per trovare il picco massimo

                synthetic_max_power_m[source].append(max_power) # Aggiunta al dizionario
            synthetic_max_power_m[source] = np.array(synthetic_max_power_m[source]) # Conversione della lista delle potenze massime sintetiche in un array

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        # Grafici log-log degli spettri di potenza sintetici mensili
        for source in synthetic_spectra_m:
            number = 0

            for spectrum in synthetic_spectra_m[source]: # Ciclo su tutti gli spettri di potenza sintetici
                tmp_len = len(spectrum['c']) // 2
                label = source if number == 0 else None # Solo alla prima curva per evitare duplicati
                axs[i].plot(spectrum['freq'][2:tmp_len], np.absolute(spectrum['c'][2:tmp_len]) ** 2, alpha=0.01, color=colors[i], label=label)
                number += 1

            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_xlabel(r'Frequenza [giorni$^{-1}$]')
            axs[i].set_ylabel(r'$|c_k|^2$')
            axs[i].set_title(f'{number_synthetic_curves} spettri sintetici')
            axs[i].legend(loc='best')
            i += 1
        plt.suptitle('Spettri di potenza sintetici - Dati mensili')
        plt.tight_layout()
        plt.show()

            # Confronto tra le potenze massime sintetiche e quella reale
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        i = 0

        print('\nAnalisi degli spettri sintetici mensili')

        for source in synthetic_spectra_m:
            tmp_len = len(monthly_source_data[source]['c']) // 2
            real_frequencies = monthly_source_data[source]['freq'][2:tmp_len]
            real_power = np.absolute(monthly_source_data[source]['c'][2:tmp_len]) ** 2
            real_max_power = trova_picco_spettro(real_frequencies, real_power)[1] # Chiamata della funzione per trovare il picco massimo

        # Confronto tra le potenze massime sintetiche e quella reale
            synthetic_max_power = synthetic_max_power_m[source] # Potenze massime sintetiche della sorgente
            number_exceeding = np.sum(synthetic_max_power >= real_max_power) # Conteggio delle potenze massime sintetiche che superano o eguagliano quella reale
            number_synthetic = len(synthetic_max_power) # Conteggio del numero totale di potenze massime sintetiche
            probability_percentage = number_exceeding / number_synthetic * 100 # Calcolo della probabilità di ottenere un picco almeno altrettanto rilevante in percentuale

            # Print dei risultati dell'analisi dei picchi sintetici
            print('\nSorgente: {}\nPotenza massima reale: {:.2e}\n'
                  'Curve sintetiche con un picco almeno altrettanto rilevante: {} su {}\n'
                  'Probabilità di ottenere un picco almeno altrettanto rilevante: {:.1f}%'.format(
                      source, real_max_power, number_exceeding,
                      number_synthetic, probability_percentage))

            # Istogrammi che confrontano i massimi sintetici e della potenza massima reale
            axs[i].hist(synthetic_max_power, bins=5, color=colors[i], edgecolor='black', label='Massimi sintetici')
            axs[i].axvline(real_max_power, color='black', linestyle='--', label='Massimo reale')
            axs[i].set_xscale('log')
            axs[i].set_xlabel(r'Massima potenza $|c_k|^2$')
            axs[i].set_ylabel('Conteggio')
            axs[i].set_title(source)
            axs[i].legend(loc='best')
            i += 1

        plt.suptitle('Distribuzione dei massimi sintetici - Dati mensili')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
