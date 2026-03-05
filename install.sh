#! /bin/bash

# Computing lib
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip install 'jax[cuda12]==0.4.33'
pip install -U jax[cuda12]==0.4.33 chex optax
# pip install optax==0.2.4
pip install flax --no-deps
# pip install tensorflow-cpu tf-keras tensorflow-probability
pip install tensorflow tf-keras tensorflow-probability

# Datasets
pip install kaggle tensorflow-datasets
# To download the datasets, you need to have a Kaggle account and a Kaggle API key.

# plotting
pip install matplotlib seaborn

# Stats
pip install pandas scikit-learn scikit-image

# CV. Open cv lib that do numpy <2.0.0
pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86

# common libraries
pip install portal colored rich ruamel.yaml==0.17.32

# RL lib
pip install gymnasium

# Miscellaneous setup

# Install ffmpeg for rendering gifs
conda install -c conda-forge ffmpeg=6.1.1 -y # version 7 does not work

# install numpy <2 (if not already)
# pip install 'numpy<2.0.0'


