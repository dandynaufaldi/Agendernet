import numpy as np
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D
from keras.layers import Activation, Multiply, Lambda, AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K


class AgenderSSRNet(Model):
    """Soft Stagewise Regression Network

    Parameters
    ----------
    image_size : int
        size for image used as input
    stage_num : list
        list of stage number
    lambda_local : float
        local lambda
    lambda_d : float
        d lambda
    """

    def __init__(self, image_size, stage_num, lambda_local, lambda_d):
        self.input_size = image_size
        if K.image_dim_ordering() == "th":
            self.__channel_axis = 1
            self.__input_shape = (3, image_size, image_size)
        else:
            self.__channel_axis = -1
            self.__input_shape = (image_size, image_size, 3)

        self.__stage_num = stage_num
        self.__lambda_local = lambda_local
        self.__lambda_d = lambda_d

        self.__x_layer1 = None
        self.__x_layer2 = None
        self.__x_layer3 = None
        self.__x = None

        self.__s_layer1 = None
        self.__s_layer2 = None
        self.__s_layer3 = None
        self.__s = None

        inputs = Input(shape=self.__input_shape)
        self.__extraction_block(inputs)

        pred_gender = self.__classifier_block(1, 'gender')
        pred_age = self.__classifier_block(101, 'age')

        super().__init__(inputs=inputs, outputs=[pred_gender, pred_age], name='SSR_Net')

    def __extraction_block(self, inputs):
        """
        Build block to extract feature from image

        Parameters
        ----------
        inputs : keras Input layer
            Input layer to be used to receive image input
        """

        x = Conv2D(32, (3, 3))(inputs)
        x = BatchNormalization(axis=self.__channel_axis)(x)
        x = Activation('relu')(x)
        self.__x_layer1 = AveragePooling2D(2, 2)(x)
        x = Conv2D(32, (3, 3))(self.__x_layer1)
        x = BatchNormalization(axis=self.__channel_axis)(x)
        x = Activation('relu')(x)
        self.__x_layer2 = AveragePooling2D(2, 2)(x)
        x = Conv2D(32, (3, 3))(self.__x_layer2)
        x = BatchNormalization(axis=self.__channel_axis)(x)
        x = Activation('relu')(x)
        self.__x_layer3 = AveragePooling2D(2, 2)(x)
        x = Conv2D(32, (3, 3))(self.__x_layer3)
        x = BatchNormalization(axis=self.__channel_axis)(x)
        self.__x = Activation('relu')(x)
        # -------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16, (3, 3))(inputs)
        s = BatchNormalization(axis=self.__channel_axis)(s)
        s = Activation('tanh')(s)
        self.__s_layer1 = MaxPooling2D(2, 2)(s)
        s = Conv2D(16, (3, 3))(self.__s_layer1)
        s = BatchNormalization(axis=self.__channel_axis)(s)
        s = Activation('tanh')(s)
        self.__s_layer2 = MaxPooling2D(2, 2)(s)
        s = Conv2D(16, (3, 3))(self.__s_layer2)
        s = BatchNormalization(axis=self.__channel_axis)(s)
        s = Activation('tanh')(s)
        self.__s_layer3 = MaxPooling2D(2, 2)(s)
        s = Conv2D(16, (3, 3))(self.__s_layer3)
        s = BatchNormalization(axis=self.__channel_axis)(s)
        self.__s = Activation('tanh')(s)

    def __classifier_block(self, V, name):
        """
        Build classifier block to calculate regression value for prediction

        Parameters
        ----------
        V : int
            Number of prediction range to be used, e.g age:100, gender:2
        name : string
            Name of prediction output ['age', 'gender']

        Returns
        -------
        keras layer
            prediction block
        """

        s_layer4 = Conv2D(10, (1, 1), activation='relu')(self.__s)
        s_layer4 = Flatten()(s_layer4)
        s_layer4_mix = Dropout(0.2)(s_layer4)
        s_layer4_mix = Dense(units=self.__stage_num[0], activation="relu")(s_layer4_mix)

        x_layer4 = Conv2D(10, (1, 1), activation='relu')(self.__x)
        x_layer4 = Flatten()(x_layer4)
        x_layer4_mix = Dropout(0.2)(x_layer4)
        x_layer4_mix = Dense(units=self.__stage_num[0], activation="relu")(x_layer4_mix)

        feat_s1_pre = Multiply()([s_layer4, x_layer4])
        delta_s1 = Dense(1, activation='tanh', name=name+'_delta_s1')(feat_s1_pre)

        feat_s1 = Multiply()([s_layer4_mix, x_layer4_mix])
        feat_s1 = Dense(2*self.__stage_num[0], activation='relu')(feat_s1)
        pred_s1 = Dense(units=self.__stage_num[0], activation="relu", name=name+'_pred_stage1')(feat_s1)
        local_s1 = Dense(units=self.__stage_num[0], activation='tanh', name=name+'_local_delta_stage1')(feat_s1)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(10, (1, 1), activation='relu')(self.__s_layer2)
        s_layer2 = MaxPooling2D(4, 4)(s_layer2)
        s_layer2 = Flatten()(s_layer2)
        s_layer2_mix = Dropout(0.2)(s_layer2)
        s_layer2_mix = Dense(self.__stage_num[1], activation='relu')(s_layer2_mix)

        x_layer2 = Conv2D(10, (1, 1), activation='relu')(self.__x_layer2)
        x_layer2 = AveragePooling2D(4, 4)(x_layer2)
        x_layer2 = Flatten()(x_layer2)
        x_layer2_mix = Dropout(0.2)(x_layer2)
        x_layer2_mix = Dense(self.__stage_num[1], activation='relu')(x_layer2_mix)

        feat_s2_pre = Multiply()([s_layer2, x_layer2])
        delta_s2 = Dense(1, activation='tanh', name=name+'_delta_s2')(feat_s2_pre)

        feat_s2 = Multiply()([s_layer2_mix, x_layer2_mix])
        feat_s2 = Dense(2*self.__stage_num[1], activation='relu')(feat_s2)
        pred_s2 = Dense(units=self.__stage_num[1], activation="relu", name=name+'_pred_stage2')(feat_s2)
        local_s2 = Dense(units=self.__stage_num[1], activation='tanh', name=name+'_local_delta_stage2')(feat_s2)
        # -------------------------------------------------------------------------------------------------------------------------
        s_layer1 = Conv2D(10, (1, 1), activation='relu')(self.__s_layer1)
        s_layer1 = MaxPooling2D(8, 8)(s_layer1)
        s_layer1 = Flatten()(s_layer1)
        s_layer1_mix = Dropout(0.2)(s_layer1)
        s_layer1_mix = Dense(self.__stage_num[2], activation='relu')(s_layer1_mix)

        x_layer1 = Conv2D(10, (1, 1), activation='relu')(self.__x_layer1)
        x_layer1 = AveragePooling2D(8, 8)(x_layer1)
        x_layer1 = Flatten()(x_layer1)
        x_layer1_mix = Dropout(0.2)(x_layer1)
        x_layer1_mix = Dense(self.__stage_num[2], activation='relu')(x_layer1_mix)

        feat_s3_pre = Multiply()([s_layer1, x_layer1])
        delta_s3 = Dense(1, activation='tanh', name=name+'_delta_s3')(feat_s3_pre)

        feat_s3 = Multiply()([s_layer1_mix, x_layer1_mix])
        feat_s3 = Dense(2*self.__stage_num[2], activation='relu')(feat_s3)
        pred_s3 = Dense(units=self.__stage_num[2], activation="relu", name=name+'_pred_stage3')(feat_s3)
        local_s3 = Dense(units=self.__stage_num[2], activation='tanh', name=name+'_local_delta_stage3')(feat_s3)
        # -------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x, s1, s2, s3, lambda_local, lambda_d, V):
            a = x[0][:, 0]*0
            b = x[0][:, 0]*0
            c = x[0][:, 0]*0

            for i in range(0, s1):
                a = a+(i+lambda_local*x[6][:, i])*x[0][:, i]
            a = K.expand_dims(a, -1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0, s2):
                b = b+(j+lambda_local*x[7][:, j])*x[1][:, j]
            b = K.expand_dims(b, -1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0, s3):
                c = c+(k+lambda_local*x[8][:, k])*x[2][:, k]
            c = K.expand_dims(c, -1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            out = (a+b+c)*V
            return out

        pred = Lambda(SSR_module,
                      arguments={'s1': self.__stage_num[0],
                                 's2': self.__stage_num[1],
                                 's3': self.__stage_num[2],
                                 'lambda_local': self.__lambda_local, 'lambda_d': self.__lambda_d, 'V': V},
                      name=name + '_prediction')([pred_s1, pred_s2, pred_s3, delta_s1, delta_s2, delta_s3, local_s1,
                                                  local_s2, local_s3])
        return pred

    def prep_phase1(self):
        """Do nothing
        """
        pass

    def prep_phase2(self):
        """Do nothing
        """
        pass

    @staticmethod
    def decode_prediction(prediction):
        """
        Decode prediction to age and gender prediction.
        Parameters
        ----------
        prediction : list of numpy array
            Result from model prediction [gender, age]
        Return
        ----------
        gender_predicted : numpy array
            Decoded gender 1 male, 0 female
        age_predicted : numpy array
            Age from regression
        """
        gender_predicted = np.around(prediction[0]).astype('int').squeeze()
        age_predicted = prediction[1].squeeze()
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
        return data


if __name__ == '__main__':
    model = AgenderSSRNet(64, [3, 3, 3], 1.0, 1.0)
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)
