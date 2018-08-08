import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model

class AgenderNetMobileNetV2(Model):
    def __init__(self):
        self.input_size = 96
        base = MobileNetV2(
            input_shape=(96,96,3), 
            include_top=False, 
            weights=None)
        top_layer = GlobalAveragePooling2D()(base.output)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(top_layer)
        age_layer = Dense(101, activation='softmax', name='age_prediction')(top_layer)
        super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetMobileNetV2')
    
    def prep_phase1(self):
        for layer in self.layers[:132]:
            layer.trainable = False

    def prep_phase2(self):
        for layer in self.layers[78:]:
            layer.trainable = True
    
    @staticmethod
    def decode_prediction(prediction):
        gender_predicted = np.argmax(prediction[0], axis=1)
        age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return gender_predicted, age_predicted

    @staticmethod
    def prep_image(data):
        data = data.astype('float16')
        data /= 128.
        data -= 1.
        return data

if __name__ == '__main__':
    model = AgenderNetMobileNetV2()
    print(model.summary())