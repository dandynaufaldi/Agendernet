import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model


class AgenderNetMobileNetV2(Model):
    """Classification model based on MobileNetV2 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 96
        base = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=os.path.dirname(os.path.abspath(__file__))+'/weight/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
        top_layer = GlobalAveragePooling2D()(base.output)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(top_layer)
        age_layer = Dense(101, activation='softmax', name='age_prediction')(top_layer)
        super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetMobileNetV2')

    def prep_phase1(self):
        """Freeze layer from input until block_14
        """
        for layer in self.layers[:130]:
            layer.trainable = False
        for layer in self.layers[130:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until blovk_8
        """
        for layer in self.layers[:78]:
            layer.trainable = False
        for layer in self.layers[78:]:
            layer.trainable = True

    @staticmethod
    def decode_prediction(prediction):
        """
        Decode prediction to age and gender prediction.
        Use softmax regression for age and argmax for gender.
        Parameters
        ----------
        prediction : list of numpy array
            Result from model prediction [gender, age]
        Return
        ----------
        gender_predicted : numpy array
            Decoded gender 1 male, 0 female
        age_predicted : numpy array
            Age from softmax regression
        """
        gender_predicted = np.argmax(prediction[0], axis=1)
        age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return gender_predicted, age_predicted

    @staticmethod
    def prep_image(data):
        """Preproces image specific to model

        Parameters
        ----------
        data : numpy ndarray
            Array of N images to be preprocessed

        Returns
        -------
        numpy ndarray
            Array of preprocessed image
        """
        data = data.astype('float16')
        data /= 128.
        data -= 1.
        return data


if __name__ == '__main__':
    model = AgenderNetMobileNetV2()
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)
