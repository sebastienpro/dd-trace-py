FROM buildpack-deps:xenial

# add our dependencies
RUN apt-get update && apt-get install -y \
    netcat \
    python-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip install tox
