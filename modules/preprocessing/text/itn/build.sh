#!/bin/bash
#
#	OpenFST
#  
FST=openfst-1.8.2
TRX=thrax-1.3.7
if [ ! -e $FST.tar.gz ] ;then
	wget http://www.openfst.org/twiki/pub/FST/FstDownload/$FST.tar.gz
fi
if [ ! -e $TRX.tar.gz ] ; then
	wget http://www.openfst.org/twiki/pub/GRM/ThraxDownload/$TRX.tar.gz
fi
tar -xvf $FST.tar.gz
tar -xvf $TRX.tar.gz

(
cd $FST || exit 1

sudo ./configure --enable-far=true --enable-pdt=true --enable-mpdt=true
sudo make
sudo make install
)

# the following directories should also be added to /etc/ld.so.conf 
# they are exported here for the purpose of building thrax 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

#
#	Thrax
#
(
cd $TRX || exit 1

sudo ./configure
sudo make
sudo make install
)
