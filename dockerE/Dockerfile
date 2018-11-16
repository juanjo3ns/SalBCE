FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /salgan_pytorch
ADD requirements.txt /salgan_pytorch/
RUN apt update -y; apt install -y \
	libsm6 \
	libxext6 \
	libxrender-dev
RUN pip3 install -r requirements.txt
