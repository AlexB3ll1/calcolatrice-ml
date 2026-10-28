# Calcolatrice con riconoscimento di cifre manoscritte

**Tipologia:** progetto di machine learning  
**Tecnologie:** Python, Keras, NumPy, Pandas, Matplotlib, Pillow, Tkinter  
**Dataset:** MNIST

## Obiettivo

Realizzare un'applicazione dimostrativa che riconosce **due singole cifre disegnate a mano** e ne calcola somma, differenza, prodotto e quoziente. Il riconoscimento avviene tramite una rete neurale convoluzionale (CNN).

## Componenti del progetto

1. **Addestramento (`reteNeurale.py`):** lettura del dataset `mnist.npz`, normalizzazione delle immagini 28×28, trasformazione delle etichette, definizione e addestramento di una CNN con Keras.
2. **Modello (`filecheckpoint1.keras`):** file salvato del modello usato dall'interfaccia grafica.
3. **Applicazione (`machineLearning1.0.py`):** finestra Tkinter con due aree di disegno. Le immagini vengono acquisite, ridimensionate a 28×28, elaborate e inviate al modello; una schermata mostra i risultati aritmetici.

## Architettura della rete presente nei sorgenti

| Fase | Componenti |
| --- | --- |
| Input | Immagine in scala di grigi di forma `28 × 28 × 1` |
| Estrazione di caratteristiche | `Conv2D(24, 5×5)` → `AveragePooling2D` → `Conv2D(64, 5×5)` → `AveragePooling2D` |
| Classificazione | `Flatten` → `Dense(480)` → `Dropout(0.5)` → `Dense(336)` → `Dropout(0.5)` → `Dense(10, softmax)` |
| Addestramento | Ottimizzatore Adam, `categorical_crossentropy`, monitoraggio della `val_accuracy`, `EarlyStopping`, `ModelCheckpoint` |

## Struttura dei file

```text
machine learning/
├── reteNeurale.py
├── machineLearning1.0.py
├── filecheckpoint1.keras
├── mnist.npz
└── Machine Learning- powerPoint.pdf
```

## Requisiti indicativi

- Python 3 e un ambiente desktop con supporto a Tkinter e cattura schermo.
- Librerie `numpy`, `pandas`, `matplotlib`, `pillow`, `keras` e un backend compatibile (ad esempio TensorFlow).

Esempio di installazione in un ambiente virtuale:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
python -m pip install numpy pandas matplotlib pillow keras tensorflow
```

Le versioni delle dipendenze non sono fissate nei sorgenti: potrebbero essere necessari adattamenti in base alla versione installata. Su alcune distribuzioni Linux Tkinter richiede un pacchetto di sistema dedicato.

## Preparazione necessaria prima di eseguire il progetto

**Nello stato originale i due script contengono percorsi Windows assoluti**. Occorre sostituirli con percorsi relativi alla cartella del progetto. Per esempio, nell'applicazione:

```python
from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parent / 'filecheckpoint1.keras'
# poi usare load_model(MODEL_PATH)
```

In `reteNeurale.py` vanno corretti anche i percorsi di `mnist.npz` e del file di checkpoint. Inoltre lo script di training fa riferimento a `drawing_data.npy`, **non presente** nell'archivio: quella sezione va rimossa, disattivata o completata con i dati necessari.

Dopo queste correzioni, l'interfaccia può essere avviata dalla cartella del progetto:

```bash
python machineLearning1.0.py
```

### Utilizzo previsto

1. Disegnare una cifra in ciascuno dei due riquadri.
2. Premere **Confirm** per eseguire la classificazione.
3. Premere **Operations** per mostrare le quattro operazioni.
4. Usare **Clear Canvases** per cancellare il disegno.

Il progetto riconosce **una cifra per riquadro**, non numeri composti da più cifre né espressioni aritmetiche complete disegnate.

## Valutazione e limiti

- Nel codice la stessa partizione MNIST `xTest/yTest` viene utilizzata come `validation_data` e poi per misurare il risultato; per una valutazione indipendente va riservato un test set non utilizzato per la selezione dei checkpoint.
- Il sorgente calcola una percentuale di previsioni corrette, ma non ne stampa esplicitamente il valore in tutti gli ambienti. I risultati dichiarati nella presentazione non sono qui certificati da un nuovo test.
- La divisione per zero non è gestita prima di calcolare il quoziente.
- L'interfaccia usa `ImageGrab` e dimensioni di finestra fisse: il comportamento può cambiare a seconda di sistema operativo, schermo e DPI.
- Lo script di training e l'applicazione **non sono stati verificati end-to-end in questa documentazione**.

## Competenze illustrate

Pre-elaborazione di immagini, addestramento di CNN, gestione di un modello serializzato e integrazione di un sistema di classificazione in una GUI Python.
