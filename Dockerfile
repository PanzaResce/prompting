FROM nvcr.io/nvidia/pytorch:22.10-py3

# ENV Variables (required for GPU access)
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"

WORKDIR /prompting
COPY . .

RUN pip3 install -r requirements.txt 