import pandas as pd 
import os
from glob import glob

from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

image_size = (128, 128)
batch_size = 32
epoch = 1

def generators60():
    dataGenerator=ImageDataGenerator(rescale=1./255., validation_split=0.25, horizontal_flip=True)
    
    testDataGenerator=ImageDataGenerator(rescale=1./255.)
    
    trainGenerator=dataGenerator.flow_from_dataframe(
    dataframe=train_set,
    directory="./images/",
    x_col="Path",
    y_col="Target Vector",
    subset="training",
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=image_size)

    validationGenerator=dataGenerator.flow_from_dataframe(
    dataframe=train_set,
    directory="./images/",
    x_col="Path",
    y_col="Target Vector",
    subset="validation",
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=image_size)

    testGenerator=testDataGenerator.flow_from_dataframe(
    dataframe=test_set,
    directory="./images/",
    x_col="Path",
    y_col="Target Vector",
    batch_size=batch_size,
    seed=42,
    shuffle=False,
    class_mode="raw",
    target_size=image_size)
    
    return trainGenerator, validationGenerator, testGenerator

def model60():
    model = Sequential()

    model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (128,128,3)))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.2))
              
    model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.2))
              
    model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 3))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(dummyLabels), activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

# VERİ YÜKLEME VE ÖN İŞLEME
# Dataframede "Path" isimli bir sütun açılır. Dataframede bulunan tüm resimlerin tam konumları ("C:/images/resim1.png" gibi) "Path" isimli sütuna kaydedilir.
dataFrame = pd.read_csv("E:/Github/Akciger-Hastaliklari-Tespit-Sistemi/Data_Entry_2017.csv")
imgPaths = {os.path.basename(x): x for x in glob("E:/Github/Akciger-Hastaliklari-Tespit-Sistemi/images/*.png")}
dataFrame['Path'] = dataFrame['Image Index'].map(imgPaths.get)

# ONE HOT ENCODING: Target Vector oluşturmak için her bir hastalık sütun olarak açılıp 0 veya 1 ile değerlindirilir.
dummyLabels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
for label in dummyLabels:
    dataFrame[label] = dataFrame['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
    
# Target Vector oluşturulur. (Modele verilecek y değeri)
dataFrame['Target Vector'] = dataFrame.apply(lambda target: [target[dummyLabels].values], 1).map(lambda target: target[0])

# Kullanılmayacak bütün sütunlar dataframeden çıkarılır.
for col in dataFrame.columns:
    if (col != "Path") and (col != "Target Vector"):
        dataFrame.drop(col, axis = 1, inplace = True)

# Dataframe test ve eğitim olarak ikiye ayrılır.
train_set, test_set = train_test_split(dataFrame, test_size = 0.25, random_state = 1999)

# Data Generatorler oluşturulur.
trainGenerator, validationGenerator, testGenerator = generators60()

# Model oluşturulur.
model = model60()

# Model eğitilir. (Model yaramaz çıktı eğitilemiyor)
model.fit(trainGenerator, validation_data=validationGenerator, steps_per_epoch=trainGenerator.n // batch_size, epochs=epoch)