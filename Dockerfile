FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /repo
ADD requirements_docker.txt /repo/requirements.txt

RUN apt update -y && apt install ffmpeg -y
RUN python3 -m pip install -U pip
RUN python3 -m pip install --no-cache -r /repo/requirements.txt
