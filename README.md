# TorchCox

https://github.com/ilanfri/TorchCox/workflows/CI/badge.svg

A validated, reasonably fast, and easily extensible implementation of a Cox model in PyTorch.

Install using pip:  
`pip3 install git+https://github.com/ilanfri/TorchCox.git`

Usage example found in `notebooks/Torch_Cox_package_test.ipynb`

Run `pytest` to perform a unit test comparing the numerical fit value against a closed-form, analytical solution of a (Maximum Likelihood Estimation) fit of a Cox model, to ensure all is well and results are scientifically valid.
