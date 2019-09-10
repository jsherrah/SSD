#FROM pytorch/pytorch
FROM nvidia/cuda:10.1-devel-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install --assume-yes apt-utils --fix-missing

RUN apt-get install -y python3

# pip3
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python3 /tmp/get-pip.py

RUN echo "alias python=python3" >> ~/.bashrc
RUN echo "alias pip=pip3" >> ~/.bashrc

RUN apt-get install -y python3-dev

RUN apt-get install -y libopencv-dev libopencv-contrib-dev emacs tmux locate wget less --fix-missing
RUN apt-get install -y eog imagemagick gthumb

RUN pip install ipython six numpy scipy Pillow
RUN pip install matplotlib scikit-image 
RUN pip install opencv-python easydict tqdm
RUN pip install torch torchvision tensorboardX
# this is the only version of tb i could get to work
RUN pip install tensorflow==1.10.0 tensorboard==1.10.0
RUN pip install imgaug torchsummary yacs vizer



# Install cuda locally.  It's already installed, but need it there for SSD nms build.
#RUN cd /tmp && wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
#RUN cd /tmp && sh cuda_10.1.243_418.87.00_linux.run --toolkit --installpath=/usr/local/cuda-10.1 --silent

# So locate command can be used.
RUN updatedb
