from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batchSize = 60
imgSize = 224
epoch = 24

train_datagen=ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('images/train', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary')
validation_set=test_datagen.flow_from_directory('images/validation', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary')

model=Sequential()

model.add(Conv2D(input_shape=(imgSize,imgSize,3), filters=32, kernel_size=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=1024,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(units=2048,activation="relu"))

model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.7))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(training_set, steps_per_epoch=training_set.n//training_set.batch_size, epochs=epoch, validation_data=validation_set, validation_steps=validation_set.n//validation_set.batch_size)

model.save("model2")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Model 2 Accuracy - Validation Accuracy")
plt.xlabel("Epoch")
plt.plot(history.history['accuracy'], 'red', label = "Accuracy")
plt.plot(history.history['val_accuracy'], 'blue', label = "Validation Accuracy")
plt.legend()
plt.savefig("model2_acc_val_acc_history")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Model 2 Loss - Validation Loss")
plt.xlabel("Epoch")
plt.plot(history.history['loss'], 'green', label = "Loss", )
plt.plot(history.history['val_loss'], 'purple', label = "Validation Loss")
plt.legend()
plt.savefig("model2_loss_val_loss_history")