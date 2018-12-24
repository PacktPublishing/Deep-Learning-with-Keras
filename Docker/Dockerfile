#################################################################################################################
# Base Images
#################################################################################################################
FROM ubuntu:14.04

#################################################################################################################
#           ENV Setting
#################################################################################################################
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

#################################################################################################################
#           Initial Setting
#################################################################################################################
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
    build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#################################################################################################################
#           User Setting
#################################################################################################################
ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src


#################################################################################################################
#           Python Setting
#################################################################################################################
# Python
RUN conda update -y python
RUN pip install numpy six tensorflow==1.3.0 keras==2.0.8 prettytensor && \
    pip install Cython ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    conda install Pillow scikit-learn notebook pandas matplotlib seaborn pyyaml h5py nltk gensim && \
    conda install -y pyqt && \
    conda clean -yt

RUN conda install -y --channel https://conda.anaconda.org/menpo opencv3 && \
    conda clean -yt

ENV PYTHONPATH='/src/:$PYTHONPATH'

#################################################################################################################
#           NLTK DOWNLOAD
#################################################################################################################
RUN python -m nltk.downloader punkt && \
    python -m nltk.downloader -d /usr/share/nltk_data brown && \
    python -m nltk.downloader -d /usr/share/nltk_data punkt && \
    python -m nltk.downloader -d /usr/share/nltk_data treebank && \
    python -m nltk.downloader -d /usr/share/nltk_data sinica_treebank && \
    python -m nltk.downloader -d /usr/share/nltk_data hmm_treebank_pos_tagger && \
    python -m nltk.downloader -d /usr/share/nltk_data maxent_treebank_pos_tagger && \
    python -m nltk.downloader -d /usr/share/nltk_data words && \
    python -m nltk.downloader -d /usr/share/nltk_data stopwords && \
    python -m nltk.downloader -d /usr/share/nltk_data names && \
    python -m nltk.downloader -d /usr/share/nltk_data wordnet
#################################################################################################################
#           WORK Jupyter
#################################################################################################################
WORKDIR /src
# USER keras

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0