from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batchSize = 32
imgSize = 224
epoch = 23

train_datagen=ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('images/train', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')
validation_set=test_datagen.flow_from_directory('images/validation', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')

model=Sequential()

model.add(Conv2D(input_shape=(imgSize,imgSize,3), filters=64, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(units=1024,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=4, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_set, steps_per_epoch=training_set.n//training_set.batch_size, epochs=epoch, validation_data=validation_set, validation_steps=validation_set.n//validation_set.batch_size)

model.save("model1")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Model 1 Accuracy - Validation Accuracy")
plt.plot(history.history['accuracy'], 'red', label = "Accuracy")
plt.plot(history.history['val_accuracy'], 'blue', label = "Validation Accuracy")
plt.legend()
plt.savefig("model1_acc_val_acc_history")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Model 1 Loss - Validation Loss")
plt.plot(history.history['loss'], 'green', label = "Loss", )
plt.plot(history.history['val_loss'], 'purple', label = "Validation Loss")
plt.legend()
plt.savefig("model1_loss_val_loss_history")
