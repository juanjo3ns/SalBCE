# SalBCE

### INTRODUCTION

![saliency_example](https://user-images.githubusercontent.com/16901615/51334553-54a8de80-1a80-11e9-95f2-9a24f20cb4cc.png)

Saliency prediction is a topic undergoing intense study in computer vision with a broad range of applications. It consists in predicting where the attention is going to be received in an image or video by a human. Our work is based on a deep neural network called [SalGAN](https://github.com/imatge-upc/saliency-salgan-2017 "Code") ([Paper](https://arxiv.org/pdf/1701.01081.pdf "Paper")) that was trained on a static dataset and it just inference each image separately without taking into account any other image or channel. With this network we've been trying to improve the saliency metrics using techniques such as depth estimation, optical flow (implemented but not trained yet), among others.

#### Model Architecture:
![encoder-decoder_salgan](https://user-images.githubusercontent.com/16901615/51336894-f67efa00-1a85-11e9-80b4-86d85b29dc0c.png)


### INSTALLATION
To run this model you'll have to pull from [this link](https://cloud.docker.com/u/juanjo3ns/repository/docker/juanjo3ns/salgan_pytorch) or build from the Dockerfile provided in this repo.
Also clone this repo and copy the same folder structure.


![Architecture](https://user-images.githubusercontent.com/16901615/51333099-57ee9b00-1a7d-11e9-83c6-e3b003312396.png)


After that, include all the desired datasets into the DATASETS folder since it'll mounted on your docker container.
To run your container type `make run` and then to attach to the bash run `make devel`. When you want to leave and stop the container press Ctrl+d and then `make down`.

#### TRAIN
As we've been working mainly with [DHF1K](https://github.com/wenguanwang/DHF1K) we are going to show all the examples with this dataset.
To run an experiment in `src/salgan_dhf1k`, run `python train_bce.py --path_out=name_of_trained_model --depth=True --daugm==True --coord=True`. Or you can run a set of experiments with the Makefile provided, you just have to edit the file with the chosen parameters.

#### TENSORBOARD
If you want to check how is the model performance while training, you can use tensorboard. From inside the container in `/home/code` run `tensorboard --logdir=trained_models`. And then in your host, enter to localhost:6006 to check the loss functions and metrics.

#### EVALUATION
To evaluate a model you should run this script `src/evaluation/eval_dhf1k.py` and select the desired parameters as well. Parameters available: `--model` (the name that you put on --path_out), `--depth, --coord`, and also an option to save the predicted images `--save`. As a results you'll get the AUC, AUCs, NSS, CC and SIM of every video and the overall average.

This experiments have been done in a GeForce GTX 1080 with 12GB RAM.
