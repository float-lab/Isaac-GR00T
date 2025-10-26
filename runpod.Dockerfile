FROM floatlab/isaac-groot

# Runpod requirements
RUN apt-get update --yes && \
    apt-get upgrade --yes

RUN apt-get install --yes --no-install-recommends \
    ca-certificates curl dirmngr file git gpg gpg-agent inotify-tools \
    jq lsof nano nginx openssh-server rsync software-properties-common \
    sudo tmux unzip vim wget zip build-essential make cmake gfortran ffmpeg \
    libatlas-base-dev libavcodec-dev libavfilter-dev libavformat-dev libblas-dev \
    libffi-dev libhdf5-dev libgl1 libjpeg-dev liblapack-dev libpng-dev \
    libpostproc-dev libsm6 libssl-dev libswscale-dev libtiff-dev \
    libv4l-dev libx264-dev libxrender-dev libxvidcore-dev \
    cifs-utils nfs-common zstd

RUN conda install "ffmpeg==7.1.1"
RUN pip install torchcodec==0.1.0
RUN apt-get install -y nginx openssh-server rsync
RUN pip install jupyterlab ipywidgets

COPY runpod/proxy/nginx.conf /etc/nginx/nginx.conf
COPY runpod/proxy/snippets /etc/nginx/snippets
COPY runpod/proxy/readme.html /usr/share/nginx/html/readme.html
COPY --chmod=755 runpod/start.sh /

#Copy dataset -- temp soln.
COPY data/lesandwich2-groot /workspace/lerobot/float-lab/lesandwich2-groot

CMD ["/start.sh"]