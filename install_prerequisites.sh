#!/usr/bin/env bash

# -- System dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    g++ \
    git \
    ninja-build \
    wget \
    ca-certificates \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    libomp-dev

# -- Working directory
TEMP_DIR="$HOME/temp"
mkdir -p "$TEMP_DIR"

# -- CMake 3.20.2
cd "$TEMP_DIR"
CMAKE_VERSION="3.20.2"
wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz"
tar -zxvf "cmake-${CMAKE_VERSION}.tar.gz"
cd "cmake-${CMAKE_VERSION}"
./bootstrap --parallel="$(nproc)"
make -j"$(nproc)"
sudo make install
cd "$TEMP_DIR"
rm -rf "cmake-${CMAKE_VERSION}" "cmake-${CMAKE_VERSION}.tar.gz"

# -- Boost 1.89.0
cd "$TEMP_DIR"
BOOST_VERSION="1.89.0"
BOOST_VERSION_U="${BOOST_VERSION//./_}"
wget "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2"
tar xf "boost_${BOOST_VERSION_U}.tar.bz2"
cd "boost_${BOOST_VERSION_U}"
./bootstrap.sh
./b2 -j"$(nproc)"
sudo ./b2 install
cd "$TEMP_DIR"
rm -rf "boost_${BOOST_VERSION_U}" "boost_${BOOST_VERSION_U}.tar.bz2"

# -- Blaze
cd "$TEMP_DIR"
git clone https://bitbucket.org/blaze-lib/blaze.git
cd blaze

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j"$(nproc)"
sudo cmake --install build
cd "$TEMP_DIR"
rm -rf blaze

echo "All dependencies installed successfully."