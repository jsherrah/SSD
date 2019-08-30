FROM pytorch/pytorch

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install --assume-yes apt-utils --fix-missing

RUN apt-get install -y libopencv-dev libopencv-contrib-dev emacs tmux locate --fix-missing 
RUN pip install opencv-python easydict
RUN pip install six numpy scipy Pillow matplotlib scikit-image imageio Shapely
RUN pip install imgaug torchsummary yacs vizer
RUN pip install torchvision==0.2.0 tensorboardX tensorboard==1.10.0
RUN pip install tqdm opencv-python

# So locate command can be used.
RUN updatedb
