FROM ubuntu:18.04

RUN apt update
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y git
RUN apt install -y python3.7 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN pip3 install numpy-stl
RUN apt update
RUN apt install -y \
    libegl1 \
    libgl1 \
    libgomp1
RUN pip3 install open3d
RUN pip3 install bpy
RUN pip3 install natsort

RUN mkdir /home/ShapeClassify