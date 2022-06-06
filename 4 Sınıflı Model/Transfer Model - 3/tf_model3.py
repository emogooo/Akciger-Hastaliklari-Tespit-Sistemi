from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
import matplotlib.pyplot as plt

batchSize = 25
imgSize = 224
epoch = 24

train_datagen=ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('images/train', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')
validation_set=test_datagen.flow_from_directory('images/validation', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')

transfer_model = DenseNet121(weights= 'imagenet', include_top = False, input_shape=(imgSize, imgSize, 3))

for layer in transfer_model.layers[:-2]:
    layer.trainable = False

new_model = Sequential()
new_model.add(GlobalAveragePooling2D(input_shape = transfer_model.output_shape[1:], data_format=None))
new_model.add(Dense(4, activation='softmax'))

model = Model(inputs=transfer_model.input, outputs=new_model(transfer_model.output))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_set, steps_per_epoch=training_set.n//training_set.batch_size, epochs=epoch, validation_data=validation_set, validation_steps=validation_set.n//validation_set.batch_size)

model.save("tf_model3")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Transfer Model 3 Accuracy - Validation Accuracy")
plt.xlabel("Epoch")
plt.plot(history.history['accuracy'], 'red', label = "Accuracy")
plt.plot(history.history['val_accuracy'], 'blue', label = "Validation Accuracy")
plt.legend()
plt.savefig("tf_model3_acc_val_acc_history")

fig, ax = plt.subplots()
ax.set_xlabel('Epoch', loc = 'right')
plt.title("Transfer Model 3 Loss - Validation Loss")
plt.xlabel("Epoch")
plt.plot(history.history['loss'], 'green', label = "Loss", )
plt.plot(history.history['val_loss'], 'purple', label = "Validation Loss")
plt.legend()
plt.savefig("tf_model3_loss_val_loss_history")
