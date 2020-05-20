from prep_data import set_test_data
from tensorflow.keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = 'dataset'
test_data_dir = os.path.join(data_dir, 'test')

num_classes = 43

x_test, y_test = set_test_data(data_dir, test_data_dir, num_classes)

model_dir = 'models/4/model.h5'

model = load_model(model_dir)

model.evaluate(x_test, y_test)
