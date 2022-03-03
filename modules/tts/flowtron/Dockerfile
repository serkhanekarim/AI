FROM nvcr.io/nvidia/pytorch:22.02-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y ffmpeg libsndfile1 sox locales vim
RUN pip install --upgrade pip
RUN pip install -U numpy
RUN pip install librosa==0.8.0 soundfile audioread sox matplotlib Pillow inflect unidecode natsort pandas jupyter tgt srt peakutils num2words
