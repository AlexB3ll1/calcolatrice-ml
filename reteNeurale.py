import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from keras.datasets import mnist

#!python test.py

#array = np.load('array.npy')

#print(array)
data = load(r'C:\Users\user\OneDrive\programmi\macine learnign\mnist.npz')
lst = data.files
xTrain, yTrain = data['x_train'], data['y_train']
xTest, yTest = data['x_test'], data['y_test']

print("suca1")
#creazione variabili x, y del train, test prendendo i dati da mnist

#xTrain.shape

#controlliamo la dimensione dei dati x di train, 60000 dati in due vettori da 28
"""
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(xTrain[i, :])
plt.show()"""
#visualizzazione immagini per capire con cosa stiamo lavorando
print("suca2")

xTrainReshaped = xTrain.reshape(60000, 28 * 28)
xTestReshaped = xTest.reshape(10000, 28 * 28)
#creazione variabili reshaped, trasformandole in matrici
#utilizzate inizialmente per poi essere ottimizzate da xScaled
print("suca3")

xTrainScaled = xTrainReshaped / 255
xTestScaled = xTestReshaped / 255
#creazione variabili scaled, normalizzando i valori
#sostituiscono le variabili reshaped in quanto forniscano mae minore
print("suca")

np.max(xTrainScaled)
#visualizzazione valore massimo di variabile normalizzata

#converto in dymmies le variabibi
yTrainDummies = pd.get_dummies(yTrain)
yTestDummies = pd.get_dummies(yTest)

yTrainDummies.shape
print("suca")

"""#costruzione della rete neurale

"""

from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Activation, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint

xTrainTensor = xTrainScaled.reshape(60000, 28, 28, 1)
xTestTensor = xTestScaled.reshape(10000, 28, 28, 1)

model = Sequential()
#creazione del modello sequenziale

#1 stato convoluzionale
model.add(Conv2D(filters = 24, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2))) #passiamo da immagine 28*28 a 14*14

#2 stato convoluzionale
model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = "valid", activation = "relu"))
#immagine ridotta da 14*14 a 10*10
model.add(AveragePooling2D(pool_size=(2, 2))) #passiamo da 10*10 a 5*5
model.add(Flatten()) #fa passare la rete da 2D a 1D

#3 stato convoluzionale
model.add(Dense(units = 480, activation = "relu"))
model.add(Dropout(0.5))

#4 stato convoluzionale
model.add(Dense(units = 336, activation = "relu"))
model.add(Dropout(0.5))

#5 stato convoluzionale
model.add(Dense(units = 10, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#compilazione del modello, con ottimizzatore adam, e mae (Mean Absolute Error) per la metrica e la perdita di dati

earlyStoppingMonitor = EarlyStopping(patience=5, monitor='val_accuracy')
#dopo 5 epochs simili accuracy blocca il traning

ModelCheckpointCallback = ModelCheckpoint(filepath =r'C:\Users\user\OneDrive\programmi\macine learnign\versioni programmi\filecheckpoint1.keras', monitor = 'val_accuracy', mode= 'auto', save_best_only= True, verbose=1)

history = model.fit(xTrainTensor, yTrainDummies, validation_data = (xTestTensor, yTestDummies), epochs = 30, callbacks=[earlyStoppingMonitor, ModelCheckpointCallback])
#esecuzione del modello con le variabili create precedentemente e ripetuto epochs volte (10)


"""predizione dataset post allenamento"""

from keras.models import load_model

bestModel = load_model(r'C:\Users\user\OneDrive\programmi\macine learnign\versioni programmi\filecheckpoint1.keras')
y_pred = np.argmax(bestModel.predict(xTestTensor), axis = 1)
y_pred
#creazione variabile di predizione arrotondando l'otput della funzione model.predict

np.mean(y_pred == yTest)
#la percentuale di successo del nostro modello

plt.figure(figsize = (20, 20))
for i in range(90):
  plt.subplot(10, 10, i + 1)
  plt.imshow(xTest[i, :])
  plt.title("Predicted :" + str(y_pred[i]))
plt.show()

drawing_data = np.load('drawing_data.npy')

# Reshape the drawing data to 4D tensor
drawing_data = drawing_data.reshape(-1, 28, 28, 1)

# Create a new model instance
model = load_model(r'C:\Users\user\OneDrive\programmi\macine learnign\versioni programmi\filecheckpoint1.keras')

# Make predictions on the drawing data
y_pred = np.argmax(model.predict(drawing_data), axis=1)

print("Prediction result:", y_pred)

# Visualize the prediction result
plt.figure(figsize=(20, 20))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(drawing_data[i, :, :, 0], cmap='gray')
    plt.title("Predicted: " + str(y_pred[i]))
plt.show()
