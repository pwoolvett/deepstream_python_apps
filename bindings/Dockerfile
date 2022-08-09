FROM nvcr.io/nvidia/deepstream:6.1-devel

# 1 - Prerequisites

## 1.1 deepstream sdk -> base image includes deepstream sdk

## 1.2 Base dependencies [Ubuntu - 20.04]
RUN apt-get update \
  && apt-get install -y \
  python3-gi \
  python3-dev \
  python3-gst-1.0 \
  python-gi-dev \
  git \
  python-dev \
  python3 \
  python3-pip \
  python3.8-dev \
  cmake \
  g++ \
  build-essential \
  libglib2.0-dev \
  libglib2.0-dev-bin \
  libgstreamer1.0-dev \
  libtool \
  m4 \
  autoconf \
  automake \
  libgirepository1.0-dev \
  libcairo2-dev \
  \
  build-essential \
  nghttp2 \
  libnghttp2-dev \
  libssl-dev

## 1.3 Initialization of submodules
WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps
RUN git init
ADD 3rdparty 3rdparty
ADD .git/modules .git/modules
### The repository utilizes gst-python and pybind11 submodules.

# ## 1.4 Installing Gst-python
# ### Following commands ensure we add the new certificates that gst-python git server now uses:
RUN apt-get install -y \
  apt-transport-https \
  ca-certificates -y \
  && update-ca-certificates

# ### Build and install gst-python:
RUN cd 3rdparty/gst-python/ \
  && ./autogen.sh \
  && make \
  && make install

# ## 2 - Compiling the bindings
ADD bindings bindings
# ### Python bindings are compiled using CMake. Following commands provide quick cmake configurations for common compilation options:
# ### 2.1.1 Quick build (x86-ubuntu-20.04 | python 3.8 | Deepstream 6.1)
RUN cd bindings \
  && mkdir build \
  && cd build \
  && cmake .. \
    -DPYTHON_MAJOR_VERSION=3 \
    -DPYTHON_MINOR_VERSION=8 \
    -DPIP_PLATFORM=linux_x86_64 \
    -DDS_PATH=/opt/nvidia/deepstream/deepstream/ \
  && make \
    -j$(nproc)


# ## 3 - Installing the bindings
RUN pip install ./pyds-1.1.3-py3-none*.whl

ADD . .
