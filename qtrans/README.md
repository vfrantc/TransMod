# qtrans

1. Use PSNR or SSIM as a loss function (l1smoothed + perceptual loss)
   https://arxiv.org/pdf/2105.06086v1.pdf PSNR
2. Make the network quaternion (https://arxiv.org/pdf/1906.04393.pdf, https://arxiv.org/pdf/2111.09881.pdf)
3. Make attentions blocks and MLP blocks harmonic (as in HarmonicNet)
4. Use interpolation instead of upsampling and downsampling (as in MPRNet)
5. Change the normalization code (maybe adapt the module from the HINet)
6. Commutative quaternion multiplication
7. If need to immrove more - use multistage as in MPRnet, HINet, MAXIM (https://arxiv.org/pdf/2201.02973v1.pdf)
8. If that's not enough - use the recursion as in PRENet

TODO: adapt the PNSR loss
TODO: + adapt the Restormer code to be quaternion
TODO: use the training procedure from TransWeather

To install the environment:

```commandline
conda create -n qtrans python=3.7
conda activate qtrans
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops
pip install timm
pip install mmcv-full
pip install opencv-python
```

Code from: 
* https://github.com/megvii-model/HINet
* https://github.com/jeya-maria-jose/TransWeather
* https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks
* https://github.com/matej-ulicny/harmonic-networks
* https://github.com/swz30/Restormer
* https://github.com/swz30/MPRNet


<details> <summary> Image Derain - Rain13k dataset (Click to expand) </summary>

* prepare data

* ```mkdir ./datasets/Rain13k```

* download the [train](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe?usp=sharing) set and [test](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) set (refer to [MPRNet](https://github.com/swz30/MPRNet))

* it should be like

  ```
  ./datasets/
  ./datasets/Rain13k/
  ./datasets/Rain13k/train/
  ./datasets/Rain13k/train/input/
  ./datasets/Rain13k/train/target/
  ./datasets/Rain13k/test/
  ./datasets/Rain13k/test/Test100/
  ./datasets/Rain13k/test/Rain100H/
  ./datasets/Rain13k/test/Rain100L/
  ./datasets/Rain13k/test/Test2800/
  ./datasets/Rain13k/test/Test1200/
  ```