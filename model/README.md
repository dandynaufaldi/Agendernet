# Model directory

- Put weight file from release inside weight/ direcotry
- Download keras pretrained weight [MobileNetV2](https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5) [InceptionV3](https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5) and save it to weight/ directory
- For insightface model
  - Download mtcnn files from [BaiduDrive](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ) or [Dropbox](https://www.dropbox.com/s/2xq8mcao6z14e3u/gamodel-r50.zip?dl=0),then save contents into weight/mtcnn-model
  - Downloadn age-gender files from [GitHub](https://github.com/deepinsight/insightface/tree/master/deploy/mtcnn-model) and save contents into weight/model-r34-age