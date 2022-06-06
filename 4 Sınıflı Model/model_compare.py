from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model1 = load_model(('Model - 1/model1'))
model2 = load_model(('Model - 2/model2'))
model3 = load_model(('Model - 3/model3'))
model4 = load_model(('Model - 4/model4'))

tmodel1 = load_model(('Transfer Model - 1/tf_model1'))
tmodel2 = load_model(('Transfer Model - 2/tf_model2'))
tmodel3 = load_model(('Transfer Model - 3/tf_model3'))

batchSize = 50
imgSize = 224
test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('images/test', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')

loss1, acc1 = model1.evaluate(test_set)
loss2, acc2 = model2.evaluate(test_set)
loss3, acc3 = model3.evaluate(test_set)
loss4, acc4 = model4.evaluate(test_set)

tloss1, tacc1 = tmodel1.evaluate(test_set)
tloss2, tacc2 = tmodel2.evaluate(test_set)
tloss3, tacc3 = tmodel3.evaluate(test_set)

fig, ax = plt.subplots()
ax.set_xlabel('Model', loc = 'right')
ax.set_ylabel('Accuracy', loc = 'top')
plt.title("Model - Accuracy")
x = np.array(["1", "2", "3", "4", "T 1", "T 2", "T 3"])
y = np.array([acc1, acc2, acc3, acc4, tacc1, tacc2, tacc3])
plt.bar(x,y,color = "blue", width = 0.3)
plt.legend()
plt.savefig("model_acc_graph")

fig, ax = plt.subplots()
ax.set_xlabel('Model', loc = 'right')
ax.set_ylabel('Loss', loc = 'top')
plt.title("Model - Loss")
x = np.array(["1", "2", "3", "4", "T 1", "T 2", "T 3"])
y = np.array([loss1, loss2, loss3, loss4, tloss1, tloss2, tloss3])
plt.bar(x,y,color = "orange", width = 0.3)
plt.legend()
plt.savefig("model_loss_graph")