# Periodicità dei blazar

Progetto del corso di **Metodi Computazionali per la Fisica**.

Lo script analizza le curve di luce settimanali e mensili di quattro blazar:
mostra le curve di luce, calcola gli spettri di potenza tramite FFT, cerca
possibili periodicità, adatta gli spettri con una legge di potenza e valuta la significatività
dei picchi mediante curve sintetiche.

## Clonazione della repository

Per ottenere una copia completa della repository, copiare e incollare nel
terminale questo comando:

```bash
git clone https://github.com/GabrieleVirdis/Curve-di-luce-metodi-computazionali.git
```

Entrare quindi nella cartella appena creata:

```bash
cd Curve-di-luce-metodi-computazionali
```

## Esecuzione

I CSV necessari sono inclusi nella cartella `data/`. 

Le analisi di significatività possono richiedere più tempo perché generano 1000 curve sintetiche per ogni sorgente.

Per eseguire il codice, dalla cartella principale della repository usare:

```bash
python3 curve_di_luce.py --option
```

## Opzioni disponibili

Le opzioni stabiliscono quali parti dell'analisi eseguire. Ogni opzione si
applica a tutte e quattro le sorgenti e può essere selezionata da sola oppure
insieme ad altre opzioni.

Le sigle si leggono in questo modo:

- `cl`: curva di luce;
- `sp`: spettro di potenza e ricerca della periodicità;
- `pf`: *fit* dello spettro di potenza;
- `ps`: spettri sintetici e significatività del picco;
- `w`: dati settimanali;
- `m`: dati mensili.

| Opzione | Cosa seleziona | Risultato |
| --- | --- | --- |
| `--clw` | Curve di luce settimanali | Apre una figura con quattro pannelli. In ciascun pannello mostra il flusso di fotoni di una sorgente in funzione della data giuliana, con le relative barre di errore. |
| `--spw` | Spettri di potenza settimanali | Calcola la FFT e lo spettro di potenza di ogni sorgente. Nel terminale riporta il picco principale, la frequenza, il periodo in giorni e anni e il numero di cicli coperti dalle osservazioni. Apre una figura con i quattro spettri e una figura che li confronta. |
| `--pfw` | Fit degli spettri settimanali | esegue il fit di ogni spettro con la legge di potenza `P(f) = N/f^β`. Stampa nel terminale `N`, `β` e le loro incertezze; mostra inoltre lo spettro e la curva adattata. |
| `--psw` | Significatività sui dati settimanali | Per ogni sorgente genera 1000 curve sintetiche rimescolando flussi ed errori, senza cambiare i tempi. Confronta il massimo osservato con i massimi sintetici, stampa quanti lo uguagliano o superano e la corrispondente probabilità empirica. Mostra anche gli spettri sintetici e gli istogrammi dei massimi. |
| `--clm` | Curve di luce mensili | Esegue la stessa visualizzazione di `--clw`, utilizzando però i dati mensili. |
| `--spm` | Spettri di potenza mensili | Esegue la stessa analisi di `--spw`, utilizzando i dati mensili. |
| `--pfm` | Fit degli spettri mensili | Esegue lo stesso adattamento di `--pfw`, utilizzando gli spettri mensili. |
| `--psm` | Significatività sui dati mensili | Esegue la stessa procedura di `--psw`, utilizzando i dati mensili e generando 1000 curve sintetiche per sorgente. |
| `-h`, `--help` | Guida rapida | Mostra nel terminale l'elenco sintetico delle opzioni e termina il programma senza eseguire l'analisi. |

## Dati e risultati

Lo script stampa i risultati numerici nel terminale e mostra i grafici
corrispondenti alle opzioni selezionate. 
Le figure prodotte per il progetto sono organizzate in:

- [`grafici/settimanale/`](grafici/settimanale/) per l'analisi settimanale;
- [`grafici/mensile/`](grafici/mensile/) per l'analisi mensile.

I valori ottenuti sono raccolti separatamente in:

- [`risultati/risultati_settimanali.txt`](risultati/risultati_settimanali.txt);
- [`risultati/risultati_mensili.txt`](risultati/risultati_mensili.txt).



