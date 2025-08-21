# vision1

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global = 1 - precision_global
  precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis = 1)).T
  if nombres != None:
    precision_categoria.columns = nombres
  
  return {
    "Matriz de Confusión" : MC,
    "Precisión Global" : precision_global,
    "Error Global" : error_global,
    "Precisión por categoría" : precision_categoria}



pip install opencv-python

import cv2

img = cv2.imread('../../../datos/img/mapache.png')
img

plt.imshow(img, cmap = "gray")
plt.show()

#identidad

identidad = np.array(
  [[0, 0, 0, 
    0, 1, 0, 
    0, 0, 0]])

img_filtro = cv2.filter2D(img, -1, identidad)
fig, ax = plt.subplots(figsize = (12, 10))
ax.imshow(abs(img_filtro), cmap = "gray")
ax.set_title("Identidad")
plt.show()

#desenfoque
desenfoque = (1/9) * np.array(
  [[1, 1, 1,
    1, 1, 1,
    1, 1, 1]])

img_filtro = cv2.filter2D(img, -1, desenfoque)
fig, ax = plt.subplots(figsize = (12, 10))
ax.imshow(abs(img_filtro), cmap = "gray")
ax.set_title("Desenfoque")
plt.show()

#enfoque

enfoque = np.array(
  [[0, -1, 0,
    -1, 5, -1,
    0, -1, 0]])

img_filtro = cv2.filter2D(img, -1, enfoque)
fig, ax = plt.subplots(figsize = (12, 10))
ax.imshow(abs(img_filtro), cmap = "gray")
ax.set_title("Enfoque")
plt.show()

img_tam = cv2.resize(img, (64, 64))
plt.imshow(img_tam, cmap = "gray")
plt.show()

img_rec = img[50:100, 100:175]
plt.imshow(img_rec, cmap = "gray")
plt.show()


#clasificacion de imagenes

import os
import re
import cv2
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU

labels = []
imagenes = []
ruta = "../../../datos/climas/"

for carpeta in next(os.walk(ruta))[1]:
  for archivo in next(os.walk(ruta + carpeta))[2]:
    if re.search("\\.(jpg|jpeg|png|bmp|tiff)$", archivo):
      try:
        img = cv2.imread(ruta + carpeta + '/' + archivo)
        img = cv2.resize(img, (64, 64))
        imagenes.append(img)
        labels.append(carpeta)
      except:
        print("No se pudo cargar la imagen: " + archivo + " en la carpeta: " + carpeta)


X = np.array(imagenes, dtype = np.int64)
y = np.array(labels)

print(
  'Total de individuos: ', len(X),
  '\nNúmero total de salidas: ', len(np.unique(y)), 
  '\nClases de salida: ', np.unique(y))


Clases de salida:  ['Cloudy' 'Rain' 'Shine' 'Sunrise']
plt.imshow(X[0])
plt.show()


plt.imshow(X[800])
plt.show()

X = X / 255


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.95, random_state = 0)


modelo_clima = Sequential()

modelo_clima.add(
  Conv2D(64, kernel_size = (3, 3), activation = 'linear',
         padding = 'same', input_shape = (64, 64, 3)))

modelo_clima.add(LeakyReLU(alpha = 0.1))
modelo_clima.add(MaxPooling2D((2, 2), padding = 'same'))
modelo_clima.add(Dropout(0.5))
 
modelo_clima.add(Flatten())
modelo_clima.add(Dense(32, activation = 'linear'))
modelo_clima.add(LeakyReLU(alpha = 0.1))
modelo_clima.add(Dropout(0.5))

modelo_clima.add(Dense(4, activation = 'softmax'))
 
modelo_clima.summary()

modelo_clima.compile(loss = "categorical_crossentropy",
                     optimizer = "adam", metrics = 'accuracy')

modelo_clima.fit(x_train, y_train, batch_size = 64,
                epochs = 50, verbose = 0)


pred = modelo_clima.predict(x_test, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred

y_test = np.argmax(y_test, axis = 1)
y_test = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test, pred, labels = encoder.classes_)
indices = indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s" % (k, str(indices[k])))
Paquetes
Cargar imágenes
Preparar datos
Modelado
Predicción
Evaluación
Guardar modelo
Predicción de individuos nuevos
modelo_clima = Sequential()

modelo_clima.add(
  Conv2D(64, kernel_size = (3, 3), activation = 'linear',
         padding = 'same', input_shape = (64, 64, 3)))

modelo_clima.add(LeakyReLU(alpha = 0.1))
modelo_clima.add(MaxPooling2D((2, 2), padding = 'same'))
modelo_clima.add(Dropout(0.5))
 
modelo_clima.add(Flatten())
modelo_clima.add(Dense(32, activation = 'linear'))
modelo_clima.add(LeakyReLU(alpha = 0.1))
modelo_clima.add(Dropout(0.5))

modelo_clima.add(Dense(4, activation = 'softmax'))

modelo_clima.compile(loss = "categorical_crossentropy",
                     optimizer = "adam", metrics = 'accuracy')
                     
modelo_clima.fit(X, y, batch_size = 64,
                epochs = 50, verbose = 0)
                
modelo_clima.save("cnn_clima.h5py")

from tensorflow.keras.models import load_model

modelo_clima = load_model('cnn_clima.h5py')

Cargar individuos nuevos
from urllib.request import Request, urlopen

def cargar_imagen_url(url):
  req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
  response = urlopen(req)
  arr = np.asarray(bytearray(response.read()), dtype = np.uint8)
  img = cv2.imdecode(arr, -1)
  return img

from urllib.request import Request, urlopen

img_1 = cargar_imagen_url('https://images.pexels.com/photos/1870259/pexels-photo-1870259.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500')
img_2 = cargar_imagen_url('https://parquesalegres.org/wp-content/uploads/2018/06/lluvia-4-1023x767.jpg')
img_3 = cargar_imagen_url('https://www.surfertoday.com/images/jamp/page/sunrisesunsettime.jpg')
img_4 = cargar_imagen_url('https://us.123rf.com/450wm/candy18/candy181801/candy18180100138/94577205-panorama-de-asfalto-de-carreteras-en-el-campo-en-un-d%C3%ADa-soleado-de-primavera-ruta-en-el-bello-paisaj.jpg?ver=6')

imgs_nuevas = [img_1, img_2, img_3, img_4]

fig, ax = plt.subplots(1, 4, figsize = (6, 6), dpi=100)
for i, axi in enumerate(ax.flat):
  no_print = axi.imshow(imgs_nuevas[i])
  no_print = axi.set(xticks = [], yticks = [])
plt.show()


#Transformar datos
imgs_nuevas = [cv2.resize(img, (64, 64)) for img in imgs_nuevas]
imgs_nuevas = np.array(imgs_nuevas, dtype = np.int64)
imgs_nuevas = imgs_nuevas / 255

#Predecir
pred = modelo_clima.predict(imgs_nuevas, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred
