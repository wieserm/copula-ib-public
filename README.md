# Learning Sparse Latent Representations with the Deep Copula IB

This repository includes a basic implementation of our paper [Learning Sparse Latent Representations with the Deep Copula IB](https://arxiv.org/pdf/1804.06216.pdf).

## Installation

The code was tested under Python 2.7.12. To run the code, it is necessary to install the following dependencies from the requirements file.

Open a new terminal and create a virtualenv:
```
mkdir copulaib
cd copulaib
git clone https://github.com/wieserm/copula-ib-public.git
cd ..
virtualenv copulaib/paper
```

Activate the environment:
```
source copulaib/paper/bin/activate
```
Install the dependencies:
```
pip install -r copulaib/copula-ib-public/requirements.txt
```

## Run the code

To run the code for the artificial experiment please execute the following command:

```
python Main.py
```

The result is an information curve plot with the used latent dimensions.

## Reference

The paper "Learning Sparse Latent Representations with the Deep Copula IB" has been accepted to ICLR 2018. If you like it please cite us.

```
@ARTICLE{WieczorekWieser,
   author = {{Wieczorek}, A. and {Wieser}, M. and {Murezzan}, D. and {Roth}, V.},
   title = "{Learning Sparse Latent Representations with the Deep Copula Information Bottleneck}",
   journal = {ArXiv e-prints},
   eprint = {1804.06216},
   year = 2018,
   month = apr,
}
```


