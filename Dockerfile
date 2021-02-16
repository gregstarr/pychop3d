# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.134.1/containers/python-3/.devcontainer/base.Dockerfile
ARG VARIANT="3.7"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.


# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends wget libspatialindex-dev libglu1 libxrender1 libxtst6 libxi6 \
    && wget https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2 \
    && tar xjf blender-2.79b-linux-glibc219-x86_64.tar.bz2 \
    && rm -rf blender-2.79b-linux-glibc219-x86_64.tar.bz2 \
    && mv blender-2.79b-linux-glibc219-x86_64 /usr/local \
    && echo export PATH=/usr/local/blender-2.79b-linux-glibc219-x86_64:$PATH >> ~/.bashrc \
    && bash -c "source ~/.bashrc"

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp