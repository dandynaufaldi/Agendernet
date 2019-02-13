import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.utils import plot_model


class AgenderNetInceptionV3(Model):
    """Classification model based on InceptionV3 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 140
        base = InceptionV3(
            input_shape=(140, 140, 3),
            include_top=False,
            weights=os.path.dirname(os.path.abspath(__file__))+'/weight/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        top_layer = GlobalAveragePooling2D(name='avg_pool')(base.output)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(top_layer)
        age_layer = Dense(101, activation='softmax', name='age_prediction')(top_layer)
        super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetInceptionV3')

    def prep_phase1(self):
        """Freeze layer from input until mixed10 (before last GlobalAveragePooling2D)
        """
        for layer in self.layers[:311]:
            layer.trainable = False
        for layer in self.layers[311:]:
            layer.trainable = True

    def prep_phase2(self):
        """Freeze layer from input until mixed5
        """
        for layer in self.layers[:165]:
            layer.trainable = False
        for layer in self.layers[165:]:
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
        data /= 127.5
        data -= 1.
        return data


if __name__ == '__main__':
    model = AgenderNetInceptionV3()
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)
