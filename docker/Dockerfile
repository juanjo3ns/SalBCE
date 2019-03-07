FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /home
ADD requirements.txt .
RUN apt update -y; apt install -y \
	libsm6 \
	libxext6 \
	libxrender-dev
RUN pip3 install -r requirements.txt
