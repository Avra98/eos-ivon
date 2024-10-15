## Variational Learning at Egde of Stability

This repository contains source code for experimenting Edge of stability behaviour for IVON.



The structure of this README is:
1. [Preliminaries](#anchors-in-markdown)
2. [Quick start](#quick-start)
3. [Complete documentation](#complete-documentation)

### Preliminaries

To run the code, you need to set two environment variables:
1. Set the `DATASETS` environment variable to a directory where datasets will be stored.
 For example: `export DATASET="/my/directory/datasets"`.
2. Set the `RESULTS` environment variable to a directory where results will be stored.
 For example: `export RESULTS="/my/directory/results"`.

### Quick start

To run the EOS dynamics for GD, please run 

```
python src/gd.py --dataset="cifar10-10k" --arch="fc-tanh"  --loss="mse"  --lr=0.05 -max_steps=10000 --neigs=2  --eig_freq=50
```

To run EOS dynamics for fixed covariance IVON run

```
python src/gd.py --dataset="cifar10-10k" --arch="fc-tanh"  --loss="mse"  --lr=0.05 -max_steps=10000 --neigs=2  --eig_freq=50 --opt="ivon" --beta2=1.0 --h0=0.7 
```

### Acknowledgement

This repository is built on top of [this implementation](https://github.com/locuslab/edge-of-stability), which is based on the paper [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://openreview.net/forum?id=jh-rTtvkGeM).
