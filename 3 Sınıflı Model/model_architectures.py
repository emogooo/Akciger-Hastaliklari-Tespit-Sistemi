from keras.models import load_model
from keras.utils.vis_utils import plot_model

model1 = load_model(('Model - 1/model1'))
model2 = load_model(('Model - 2/model2'))
model3 = load_model(('Model - 3/model3'))

tmodel1 = load_model(('Transfer Model - 1/tf_model1'))

plot_model(model1, to_file='Model - 1/model1_architecture.png', show_shapes=True, show_layer_names=True)
plot_model(model2, to_file='Model - 2/model2_architecture.png', show_shapes=True, show_layer_names=True)
plot_model(model3, to_file='Model - 3/model3_architecture.png', show_shapes=True, show_layer_names=True)

plot_model(tmodel1, to_file='Transfer Model - 1/tmodel1_architecture.png', show_shapes=True, show_layer_names=True)
