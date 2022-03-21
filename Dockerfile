FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get -y update
RUN apt-get -y install wget 
RUN apt-get -y install libspatialindex-dev 
RUN apt-get -y install libglu1
RUN apt-get -y install libxrender1
RUN apt-get -y install libxtst6
RUN apt-get -y install libxi6

COPY requirements.txt .
COPY blender-2.79b-linux-glibc219-x86_64.tar.bz2 .

RUN tar xjf blender-2.79b-linux-glibc219-x86_64.tar.bz2
RUN export PATH=/usr/src/app/blender-2.79b-linux-glibc219-x86_64:$PATH
RUN export PYTHONPATH=/usr/src/app/pychop3d

RUN python -m pip install --upgrade --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pip
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir pybind11
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir meshpy
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt
