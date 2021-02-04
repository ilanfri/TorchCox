![status](https://github.com/ilanfri/TorchCox/workflows/CI/badge.svg "CI build status")

# TorchCox


A validated, reasonably fast, and easily extensible implementation of a Cox model in PyTorch.

**The goal of this package -and hope of the author- is that this bare-bones code is simple, understandable, fast, and trustworthy enough for you to use it to build your own extensions on top of it.**

Install using pip:  
`pip install git+https://github.com/ilanfri/TorchCox.git`

Usage example found in `notebooks/Torch_Cox_package_test.ipynb`

Run `pytest` to perform a unit test comparing the numerical fit value against a closed-form, analytical solution of a (Maximum Likelihood Estimation) fit of a Cox model, to ensure all is well and results are scientifically valid.

The CI (Continuous Integration) badge just above shows whether this package both compiles correctly and matches the closed-form solution. If its value is 'passing', all is well and you can trust this version.  

This is by no means the first implementation of a Cox model in PyTorch, ones with more complete functionality exist (e.g. [pycox](https://github.com/havakv/pycox)). It is rather a bare-bones implementation by the author for personal use and research in extending the joint fitting of parameters enabled by differentiable programming to certain extensions of Cox models which I have in mind (to be shared later).  

**New developments on this repo will be shared [by me on Twitter](https://twitter.com/irfnali1).**


## Validation

**The validation can be examined in the notebook found in `notebooks/Validation.ipynb`.**

The regression coefficients of this code have been validated against a closed-form solution on a simple synthetic dataset, and against the [R survival package](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/00Index.html) which is a standard tool of the trade.  

The resulting unit test is in the `tests/test_TorchCox.py` which the user can run anytime by runing `pytest` from bash from within the package directory, and which is run automatically as part of the Continuous Integration anytime I push changes to this GitHub repository, resulting in the 'CI passing' badge above, which indicates the package both installs properly (with Python 3.8) and also the comparison against the closed-form result matches to **5 decimal places**.


## Advantages of a Cox model implemented in a Differentiable Programming language

Blazingly fast and reliable implementations with a lot more functionality (like the R package `survival`) already exist, why would anyone bother implementing this in a Differentiable Programming language like PyTorch? Some reasons are:

- Extensibility: changes to loss function or optimisation algorithm are often one-line changes
- Scalability: functionality to deploy across multiple CPUs or GPUs is often built in or easy to include
- Mobile deployment: if relevant, models can be deployed on mobile devices (Android or iOS)
- Automatic differentiation: computing confidence intervals originally involved computing second derivatives by hand and implementing the result in the code, with differentiable programming simply changing the loss is sufficient, the computation of second derivatives is automatic (provided loss is twice-differentiable, obviously)
- Ecosystem: integration with existing PyTorch libraries (see https://pytorch.org/ecosystem/) to add all sorts of functionality should be straightforward 


## Challenge: Data representation

Differentiable Programming frameworks like PyTorch are designed and optimised to compute on multi-way arrays (I'll twist my own arm here and call them _tensors_ for once, though these are as much tensors as any two-tuple is a complex number, but don't get me started), which means that getting them to compute a Cox model efficiently involves encoding survival analysis data in tensors.

The way this was done was using what for the sake of having a name to refer to it by, as _staircase encoding_, which I describe in `notebooks/Staircase_encoding.ipynb`.


## How it works

How the staircase encoding data representation is used together with PyTorch's tensor methods and autograd to efficiently fit a Cox model is shown in `notebooks/How_it_works.ipynb`.


## Data

The datasets included in `data` are modified versions (due to an issue where Torch `float()` precision truncated event times and introduced spurious tied events in a handful of cases, which spoilt comparisons against R) of the `ovarian` dataset available through the [R survival package](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/ovarian.html), and originally from:  

Edmunson, J.H., Fleming, T.R., Decker, D.G., Malkasian, G.D., Jefferies, J.A., Webb, M.J., and Kvols, L.K., Different Chemotherapeutic Sensitivities and Host Factors Affecting Prognosis in Advanced Ovarian Carcinoma vs. Minimal Residual Disease. Cancer Treatment Reports, 63:241-47, 1979.



## Reference

If you do use the present codebase in your work would hugely appreciate it if you cite it as below:  
```
@Misc{torchcox,
  author = {Fridman Rojas, Ilan},
  title = {{TorchCox}: A validated, reasonably fast, and easily extensible implementation of a Cox model in PyTorch.},
  year = {2021},
  url = "https://github.com/ilanfri/TorchCox"
}
```