import keras2onnx
import onnx
from keras.models import load_model

model = load_model('./weights/20220328_myself_model.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = './weights/20220328_model.onnx'
onnx.save_model(onnx_model, temp_model_file)
