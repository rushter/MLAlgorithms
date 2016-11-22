FROM buildpack-deps:trusty

RUN mkdir -p /var/app
WORKDIR /var/app
COPY . /var/app

# update apt cache
# install python & python-dev
# clean up
# clean up
# get the pip installer
# install pip
# clean up
# install required pip packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends python python-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install scipy numpy && \
    pip install .
