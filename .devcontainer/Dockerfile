FROM nvcr.io/nvidia/pytorch:24.04-py3

#Para aceptar las instalaciones predeterminadas de forma automatica
ARG DEBIAN_FRONTEND=noninteractive

USER root
### El UID y GID del host deben ser los mismos que el user del contenedor para tener los mismos permisos
# RUN groupadd --gid 1001 pytorch-group && \
    # useradd -rm -d /home/pytorch -s /bin/bash -g pytorch-group -u 1000 pytorch


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#Se agrega al grupo conda para tener permisos del entorno conda
#RUN usermod -a -G conda pytorch

USER pytorch

WORKDIR /home/pytorch/data

RUN pip install \
        plotly \
		ujson \
        jupyterlab-language-pack-es-ES \
		dash \
		dash-core-components \
		dash-html-components \
		networkx \
		pycocotools \
		PyYAML \
		tensorboard \
		opencv-python==4.7.0.72
    
EXPOSE 8888