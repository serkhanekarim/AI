FROM ubuntu:latest
RUN apt-get update && apt-get install -y file g++ make sudo gawk tar wget python python3
COPY . /app/
WORKDIR /app
RUN ./build.sh
ENV LD_LIBRARY_PATH=/usr/local/lib
#WORKDIR /app/fst-thrax-grammars-inv/fr-FR/
#RUN cd /app/fst-thrax-grammars-inv/fr-FR/
#RUN thraxmakedep itn.grm
#RUN make
#CMD thraxrewrite-tester --far=itn.far --rules=MAIN1,MAIN2
