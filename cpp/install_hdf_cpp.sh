apt-get install libhdf5-dev
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.1.tar.gz
tar -xvf hdf5-1.10.1.tar.gz
cd hdf5-1.10.1
./configure --enable-cxx --prefix /usr/local/hdf5 && make && make check && sudo make install