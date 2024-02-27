ARG CUDA_VERSION=11.1.1
ARG OS_VERSION=20.04
# pull a prebuilt image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

# setup timezone
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y sudo
RUN sudo apt install -y git\
                        python3.8\
                        python3-pip\
                        wget\
                        python3-tk

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglib2.0-0 && apt-get install -y libgl1-mesa-glx

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /workspace

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

WORKDIR /DAB-DETR




###########
# docker run -it --gpus all --net=host --ipc=host --pid=host --mount type=bind,src="$(pwd)/",target="/DAB-DETR/" --shm-size 32G detr_new