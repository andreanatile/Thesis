features_extraction.py
    in questo file sono presenti tutte le funzioni utili all'estrazione delle feature.
        - features_extraction è una funzione che prendendo come input il segnale, la 
        frequenza di campionamento a cui è stato misurato, la lunghezza dei segmenti 
        e la percentuale di sovrappizione dei segmenti, ritorna un dataframe contentente
        per ogni segmento le feature estratte.

        - Combining_features è una funzione che prendendo come input 3 segnali ritorna un 
        dataframe omnicomprensivo di tutte le feature calcolate per ogni segmento di ogni segnale
        inserito. Il senso di questa funzione è quello di avere un modo compatto per estrarre
        tutte le feature utili all'allenamento del rilevatore in un uncica riga di codice, seguendo il
        paradigma della modularità.

        - feature_extraction_from_segments ha lo stesso funzionamento di features_extraction,
        diverge che il segnale in ingresso è direttamente segmentato, utile per l'estrazione delle 
        feature dai segmenti direttamente filtrati

        -Combining_features_from_segments ha lo stesso funzionamento di Combining_features, soltato
        che in ingeresso i 3 segnali sono stati già segmentati, utile per l'estrazione delle 
        feature dai segmenti direttamente filtrati
    
notch_filter.py
    in questo file sono presenti le funzioni per creare il filtro adattivo stoppabanda.

        -notch_filter_segment è una funzione che prende in ingresso il risultato
        della FFT di un segmento, l'indice dell'array della banda di frequenza con energia più
        alta del segmento precedente, il valore dell'energia della banda a maggiore energia del segmento precedente,
        la frequenza di campionamento e la sovrappizione desiderata tra le bande di frquenza.
        confronta la massima energia del segmento corrente con quella del segmento precedente, nel 
        caso c'è uno scarto di al più due indici tra le bande di frequenza e uno scarto al più del 40% tra i valori di 
        energia massima, il segnale viene filtrato secondo un filtro notch. in caso contrario non viene filtrato.
        in ritorno da il segmento filtrato in caso è stato filtrato oppure il segmento dato in input.

        -notch_filter_data prende in input il segnale, la lunghezza dei segmentie la frequenza di campionamento
        lo segmente e ad ogni segmento applica la funzione notch_filter_segment. infine ritorna la 
        lista di segmenti filtrati. utile per filtrare in modo compatto il segnale in un'unica riga di codice, applicando
        il paradigma della modularità.

labelling.py
    in questo file sono presenti delle funzioni per convertire il file text del software Tero Subliter
    utilizzato per trascrivere la registrazione audio utilizzata per il labelling, in un dataframe da poter 
    fare il merge con le feature calcolate.

        -Extract_Labels_fromTeroSubliter prende in input la stringa del file di Tero Subliter e ritorna
        un dataframe contentente l'inizio e la fine dell'anomalia con il valore della classe dell'anomalia.

        - Merge_Feature_Label è una funzione che prendendo in input il dataframe delle feature 
        e il dataframe delle lables, associa ai segmenti in cui l'anomalia è inclusa all'interno dell'inizio e la fine del segmento
        stesso. In tutti i segmenti privi di anomalia alla colonna Anomaly viene assegnato il 
        valore 'ok'.

plot_notchfilter1.py e plot_notchfilter2.py
    in questi file vengono plottati i grafici per la spiegazione del funzionamento del filtro 
    adattivo.

speed_dependency.py
    file utilizzato per la demodulazione del segnale in modo da eliminare la dipendenza
    della velocità nei segnali. Non è stato utilizzata questa tecnica poichè è stata riscontrata 
    una difficoltà nella sua comprensione.


