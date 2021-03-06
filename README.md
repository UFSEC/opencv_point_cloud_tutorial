# Point Clouds from RGB and Depth videos #

## Setup ##

### Pro Mode ###
Use Docker to handle all of your dependencies :)
* Install docker: `sudo apt-get install docker`
* Build the docker container: `docker build -t sec_pc_tutorial .`
* Run the Docker container: `./runDocker sec_pc_tutorial`

### Less Pro Mode ###
Install all these dependencies yourself (already out of date :/)

#### Pre-OpenCV Dependencies ####
OpenCV 3.0 with Python bindings has some required dependencies. You can install them with this these commands.
`sudo apt-get install -y build-essential`
`sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev`
`sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev`

#### How to Build OpenCV 3.0 ####

Get the Opencv 3.0 Release Candidate 1 from Github

`git clone https://github.com/Itseez/opencv.git`

`git checkout 3.0.0-rc1`

Build from source (this will take a while)

`mkdir build install && cd build`
`cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=../install ..`
`make -j$(($(nproc)+1))`

Add it to your PATH
`export LD_LIBRARY_PATH=<path/to/opencv/install/dir>/lib:$LD_LIBRARY_PATH`
`export PATH=<path/to/opencv/install/div>/bin:$PATH`

## How to Run these Programs ##

* Programs 1-3: `python PROGRAM_NAME`

* Programs 4-5:
```
mkdir build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make && make install
cd ../install/bin/
./PROGRAM_NAME
```
