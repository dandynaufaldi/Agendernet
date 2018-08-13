# Age and Gender Predictor
**Update from old [repo](https://github.com/dandynaufaldi/Age-Gender-Predictor)**

## Environment Setup
### Docker with GPU support
- Get [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Pull my [docker image](https://hub.docker.com/r/dandynaufaldi/tf-keras-cuda9-cudnn7/) by `docker pull dandynaufaldi/tf-keras-cuda9-cudnn7`, the Dockerfile will come in future  
### Manually
Using python 3.5, core libraries are :
- [dlib](https://github.com/davisking/dlib)
- tensorflow
- keras
- opencv
- mxnet

## Datasets
Datasets are saved in data/ directory
- [IMDB-Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
- [UTKFace](https://susanqq.github.io/UTKFace/)
- [FGNET](http://yanweifu.github.io/FG_NET_data/index.html)

## Preprocess


## Train


## Evaluation


## Test


## References and Acknowledgments
This project is part of my internship program at [Nodeflux](https://nodeflux.io/) as data scientist from July - August, 2018
1. [Rothe R, Timofte R, Van Gool L. Dex: Deep expectation of apparent age from a single image[C]//Proceedings of the IEEE International Conference on Computer Vision Workshops. 2015: 10-15.](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf)
2. [Rothe R, Timofte R, Van Gool L. Deep expectation of real and apparent age from a single image without facial landmarks[J]. International Journal of Computer Vision, 2016: 1-14.](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf)
3. [[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation](https://github.com/shamangary/SSR-Net)
4. [yu4u/age-gender-estimation Keras implementation of a CNN network for age and gender estimation](https://github.com/yu4u/age-gender-estimation)
5. [deepinsight/insightface Face Recognition Project on MXNet](https://github.com/deepinsight/insightface)
6. [abewley/sort Simple, online, and realtime tracking of multiple objects in a video sequence](https://github.com/abewley/sort)