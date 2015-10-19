FROM ubuntu:14.04
MAINTAINER Nick Tardif <nicktardif@gmail.com>

RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential

# OpenCV Dependencies
RUN apt-get install -y cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev
RUN apt-get install -y python-dev \
    python-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer
USER developer
WORKDIR /home/developer/

# Download, install, and build OpenCV
RUN git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    git checkout 3.0.0-rc1 && \
    mkdir build install &&  \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../install ..
RUN cd opencv/build && make -j$(($(nproc)+1)) && make install

# Set environment variables to use OpenCV
ENV LD_LIBRARY_PATH /home/developer/opencv/install/lib:$LD_LIBRARY_PATH
ENV PATH /home/developer/opencv/install/bin:$PATH

# Other things to install
RUN sudo apt-get install -y feh \
    wget

RUN git clone https://github.com/UFSEC/opencv_point_cloud_tutorial.git
CMD /bin/bash
