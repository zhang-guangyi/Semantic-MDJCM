# PyTorch implementation of [From Analog to Digital: Multi-Order Digital Joint Coding-Modulation for Semantic Communication](https://ieeexplore.ieee.org/abstract/document/10778620)

This repository is built upon [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22), thanks very much!

We would gradually upload the full-version of the implementation.

## Citation
``` bash
@ARTICLE{zhang2024MDJCM,
  author={Zhang, Guangyi and Yang, Pujing and Cai, Yunlong and Hu, Qiyu and Yu, Guanding},
  journal={IEEE Transactions on Communications}, 
  title={From Analog to Digital: Multi-Order Digital Joint Coding-Modulation for Semantic Communication}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Entropy;Modulation;Training;Source coding;Encoding;Symbols;Quantization (signal);Transforms;Image coding;Vectors;Digital modulation;multi-order modulation;joint source-channel coding;semantic communications},
  doi={10.1109/TCOMM.2024.3511949}}
```


## Usage
### Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/MDJCM.git
cd MDJCM/
```

### Requirements
`Python 3.9.12` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page (https://pytorch.org/get-started/previous-versions/).

## Pretrained Models
- Download [MDJCM-ckpt](https://pan.baidu.com/s/1-xiEIH6eG2xnsOPDvlbo3Q?pwd=94yp) and put them into ./ckpt folder.

## Usage
1. Example of train the MDJCM-A model:
```bash
bash train.sh
```

2. Example of test the MDJCM-A model:
- Run test.sh
```bash
bash test.sh
``` 
