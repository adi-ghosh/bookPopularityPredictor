from tensorflow import keras


model = keras.models.load_model('model/model1.keras')
model.predict()