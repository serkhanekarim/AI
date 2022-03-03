FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y file g++ make sudo gawk tar wget
COPY . /app/
WORKDIR /app
RUN sudo ./build.sh
ENV LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /app/fst-thrax-grammars-inv/fr-FR/
RUN cd /app/fst-thrax-grammars-inv/fr-FR/
RUN thraxmakedep itn.grm
RUN make
CMD thraxrewrite-tester --far=itn.far --rules=MAIN1,MAIN2
