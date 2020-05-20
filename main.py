from prep_data import set_train_data
from build_model import build_model, train_model
from visualize import visualize_history
import os


data_dir = 'dataset'
train_data_dir = os.path.join(data_dir, 'train')

num_classes = 43

x_train, y_train, x_val, y_val = set_train_data(
    train_data_dir, num_classes)

input_shape = x_train.shape[1:]

model = build_model(input_shape, num_classes)
model, history = train_model(model, x_train, y_train, x_val, y_val)

visualize_history(history)
