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
