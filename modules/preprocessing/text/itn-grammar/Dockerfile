FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y file g++ make sudo gawk tar wget python python3
COPY . /app/
WORKDIR /app
RUN ./build.sh
ENV LD_LIBRARY_PATH=/usr/local/lib
