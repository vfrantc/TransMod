conda create -n retinex python=3.8
conda activate retinex
conda install pytorch torchvision -c pytorch
conda install -c conda-forge opencv
python3 train.py --data_dir="."
