import os
import numpy as np
import mxnet as mx
import cv2
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans


def get_model(ctx: mx.gpu, image_size: tuple, model_str: str, layer: str):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def resize_image(image, size=140):
    """Get squared-resized image
    """
    BLACK = [0, 0, 0]
    h = image.shape[0]
    w = image.shape[1]
    if w < h:
        border = h-w
        image = cv2.copyMakeBorder(image, 0, 0, border, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        border = w-h
        image = cv2.copyMakeBorder(image, border, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized


class InsightFace:
    def __init__(self):
        ctx = mx.gpu(0)
        self.image_size = (112, 112)
        self.ga_model = get_model(ctx, self.image_size, os.path.dirname(
            __file__)+'/weight/model-r34-age/model,0', 'fc1')
        self.threshold = 1.24
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'weight/mtcnn-model')
        detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
                                 accurate_landmark=True, threshold=[0.0, 0.0, 0.2])
        self.detector = detector

    def prep_image(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=0)
        if ret is None:  # return resized image
            res = resize_image(face_img, 112)
            nimg = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            return aligned
        bbox, points = ret
        if bbox.shape[0] == 0:  # return resized image
            res = resize_image(face_img, 112)
            nimg = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            return aligned
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        nimg = preprocess(face_img, bbox, points, image_size='112,112')
        if nimg is None:
            nimg = resize_image(face_img, 112)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def predict(self, aligned: np.ndarray, **kwargs):
        batch_size = kwargs.get('batch_size', 128)
        input_blob = aligned
        data = mx.nd.array(input_blob)
        data_iter = mx.io.NDArrayIter(data, batch_size=batch_size)
        genders = []
        ages = []
        for db in data_iter:
            self.ga_model.forward(db, is_train=False)
            raw_output = self.ga_model.get_outputs()
            output = mx.nd.stack(*raw_output).asnumpy().squeeze()
            if (len(output.shape) == 1):  # if only 1 input and 1 output
                output = np.expand_dims(output, axis=0)
            g = output[:, 0:2].reshape((db.data[0].shape[0], 1, 2))
            gender = np.argmax(g, axis=2).flatten()
            a = output[:, 2:202].reshape((db.data[0].shape[0], 100, 2))
            a = np.argmax(a, axis=2)
            age = a.sum(axis=1)
            genders.append(gender)
            ages.append(age)
        genders = np.concatenate(genders).astype('int')[:len(aligned)]
        ages = np.concatenate(ages)[:len(aligned)]
        return list(genders, ages)

    @staticmethod
    def decode_prediction(prediction):
        """Decode prediction to age and gender prediction.
        """
        gender_predicted = prediction[0]
        age_predicted = prediction[1]
        return gender_predicted, age_predicted
