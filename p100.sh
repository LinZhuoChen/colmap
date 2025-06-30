FROM reg.docker.alibaba-inc.com/aii/aistudio:142-20240808101605
yum install -y boost-devel 
# ossutil64 cp -r oss://antsys-vilab/zsz/CMake ./
# cd CMake/CMake
# chmod 777 -R *
# ./configure
# make -j 32
# make install
# rm -rf /usr/local/bin/cmake
# cp bin/cmake /usr/local/bin/
# export PATH=:$(pwd)/bin:$PATH
# export CMAKE_ROOT=/usr/local/share/cmake-3.30/
# git checkout .
# mkdir build
# git checkout 3.7
cd build
pip install ninja
ossutil64 cp -r oss://antsys-vilab/zsz/eigen ./ -j 400 -u
cd eigen/eigen/build
yum install libgfortran -y
yum install gcc-gfortran -y
cmake ..
make install -j 16
cd ../../../
ossutil64 cp -r oss://antsys-vilab/zsz/flann ./ -j 20
cd flann
mkdir build
cd build
yum install lz4-devel -y
cmake ..
make install -j 16
cd ../..
yum install mesa* -y
yum install freeglut* -y
ossutil64 cp -r oss://antsys-vilab/zsz/ceres-solver ./ -j 200 -u
cd ceres-solver
mkdir build
cd build
cmake ..
make install -j 16
cd ../..
ossutil64 cp -r oss://antsys-vilab/zsz/abseil-cpp ./ -j 200 -u
cd abseil-cpp
mkdir build
cd build
cmake ..
make install -j 16
cd ../..
ossutil64 cp -r oss://antsys-vilab/zsz/cgal ./ -j 200 -u
cd cgal
mkdir build
cd build
cmake ..
make install -j 16
cd ../..
yum install -y gmp-devel -y
yum install -y \
    gflags-devel \
    glog-devel \
    glew-devel 
# mpfr
cmake .. -DCMAKE_CUDA_COMPILER:PATH=nvcc  -GNinja -DGUI_ENABLED=False      

ninja
ninja install