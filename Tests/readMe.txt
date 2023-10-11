cartella dove viene fatto l'allenamento e il test del rilevatore SVM e 
l'ottimizzazione dei iperparametri tramite il GridSearch.
In particolare:
    raw_data_extraction.py -> Script per l'allenamento del rilevatore utilizzando 
    dati provenienti da data\Training_Datasets\raw_Training_Dataset.csv, cioè utilizzando
    feature estratte da i dati grezzi, cioè non filtrati.

    notch_Test.py -> Script per l'allenamento del rilevatore utilizzando 
    dati provenienti da data\Training_Datasets\notchFiltered_Training_Dataset.csv, cioè utilizzando
    feature estratte da i dati filtrati con il filtro adattivo.