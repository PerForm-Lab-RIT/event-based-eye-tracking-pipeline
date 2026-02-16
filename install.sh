
# Before running this script, you can create a miniconda environment with:
# conda activate amin

# Install pytorch for cuda 12.6
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install pytorch for cuda13. NOTE: For some machine, the gpu driver is not updated, so it's best to stay with cuda 12 for now
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# install normal pytorch for cuda 12.8
pip install torch torchvision

pip install snntorch

pip install tonic

# Intall box2d dependencies (requires special handling)
conda install -c conda-forge swig boost-cpp -y

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Then install the requirement file
pip install -r "$SCRIPT_DIR/requirements.txt"

# Since hugginface_hub is updated continuously (and often not aligned with transformers),
#  we install it manually
# This is used to download the data from huggingface using hf download
pip install -U huggingface_hub

# Install ffmpeg for rendering gifs
conda install -c conda-forge ffmpeg=6.1.1 -y # version 7 does not work
# Install additional video utilities package after ffmpeg
pip install av torchcodec


# pip install hydra-core --upgrade
# if encounter build error because of outdated JAVA version, run
#   `sudo apt install default-jdk` to install JAVA 11 or above
# Note: For Python 3.13+, need to install older setuptools first and use --no-build-isolation
pip install "setuptools<70"  # Ensure pkg_resources is available
pip install --no-build-isolation git+https://github.com/facebookresearch/hydra.git
# pip install git+https://github.com/rxng8/hydra.git

