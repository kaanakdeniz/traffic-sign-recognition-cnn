import numpy as np
from tensorflow.keras.models import load_model
from visualize import visualize_history


# Verilerin yüklenmesi
prep_data_dir = 'preprocessed data'

x_train = np.load(prep_data_dir + '/x_train.npy')
y_train = np.load(prep_data_dir + '/y_train.npy')
x_val = np.load(prep_data_dir + '/x_val.npy')
y_val = np.load(prep_data_dir + '/y_val.npy')
x_test = np.load(prep_data_dir + '/x_test.npy')
y_test = np.load(prep_data_dir + '/y_test.npy')

# Modelin ve historynin yüklenmesi
model_dir = "models/4"

model = load_model(model_dir + '/model.h5')
history = np.load(model_dir + '/history.npy', allow_pickle='TRUE').item()

# Historynin görselleştirilmesi
visualize_history(history)

# Modelin test edilmesi
model.evaluate(x_test, y_test)
