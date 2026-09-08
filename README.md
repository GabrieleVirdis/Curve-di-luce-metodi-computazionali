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

I CSV necessari sono inclusi nella cartella `Dati/`. 

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
- `pf`: fit dello spettro di potenza;
- `ps`: spettri sintetici e significatività del picco;
- `w`: dati settimanali;
- `m`: dati mensili.

### Dati settimanali

- `--clw` mostra le curve di luce delle quattro sorgenti, con il flusso in
  funzione della data giuliana e le relative barre di errore.
- `--spw` calcola gli spettri di potenza tramite FFT. Mostra i grafici e stampa
  frequenza, potenza, periodo e numero di cicli del picco principale.
- `--pfw` esegue il fit degli spettri con la legge di potenza
  `P(f) = N/f^β`. Mostra i grafici e stampa `N`, `β` e le loro incertezze.
- `--psw` genera 1000 curve sintetiche per sorgente e confronta i loro picchi
  con quello osservato. Mostra gli spettri e gli istogrammi e stampa la
  probabilità empirica.

### Dati mensili

- `--clm` mostra le curve di luce mensili.
- `--spm` calcola e analizza gli spettri di potenza mensili.
- `--pfm` esegue il fit degli spettri mensili.
- `--psm` valuta la significatività sui dati mensili generando 1000 curve
  sintetiche per sorgente.

Per visualizzare l'elenco sintetico delle opzioni nel terminale, usare `-h`
oppure `--help`.

## Dati e risultati

Lo script stampa i risultati numerici nel terminale e mostra i grafici
corrispondenti alle opzioni selezionate. 
Le figure prodotte per il progetto sono organizzate in:

- [`Grafici/settimanale/`](Grafici/settimanale/) per l'analisi settimanale;
- [`Grafici/mensile/`](Grafici/mensile/) per l'analisi mensile.

I valori ottenuti sono raccolti separatamente in:

- [`Risultati/risultati_settimanali.txt`](Risultati/risultati_settimanali.txt);
- [`Risultati/risultati_mensili.txt`](Risultati/risultati_mensili.txt).
