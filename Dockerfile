FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get upgrade -y && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-core-dev \
    ffmpeg \
    nano \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# Headless BEV calibration UI (bev_web_calibrator.py). Forward this port:
#   docker run -p 8000:8000 ...   then open http://localhost:8000
EXPOSE 8000