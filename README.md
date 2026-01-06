# Data Driven SIMCA

The package *ddsimca* implements *Data Driven SIMCA* — a method for creating one-class classification (OCC) models (also known as *anomaly detectors* or *novelty detectors*). The theoretical background and practical examples for the DDSIMCA method are described in [this paper](https://doi.org/10.1002/cem.3556), please use it for citing this package as well. The paper is freely available to everyone via open access option. The other papers describing different theoretical aspects of the method are listed in the *Reference* section below.

The paper shows all examples using the DD-SIMCA web-application ([mda.tools/ddsimca](https://mda.tools/ddsimca)). This package implements same functionality in Python, so you can get same outcomes and similar plots. The package can be installed from [PyPI](https://pypi.org) using `pip` or any other package manager compatible with PyPI, e.g.:


```
pip install ddsimca
```

It requires `numpy`, `scipy`, `pandas`, `prcv` and `matplotlib`,  which will be automatically installed as dependencies.


## Getting started

Use Jupyter notebook [demo.ipynb](https://github.com/svkucheryavski/ddsimca-py/blob/main/demo/demo.ipynb) or Markdown document [demo.md](https://github.com/svkucheryavski/ddsimca-py/blob/main/demo/demo.md) in order to get started. To run the examples from this notebook you need to download zip file with datasets (it is also used for illustration of the method in the paper). Here is [direct link](https://mda.tools/ddsimca/Oregano.zip) to the archive.

Simply download the dataset and unzip it to your working directory, where you have the notebook or Markdown document, and follow the guidelines. The dataset can be downloaded from GitHub as well.

## Releases

**1.0.3** (6/1/2026)
* small improvements to code.
* better documentation text.

**1.0.2** (1/1/2026)
* initial release



## References

1. S. Kucheryavskiy, O. Rodionova, A. Pomerantsev. *A comprehensive tutorial on Data-Driven SIMCA: Theory and implementation in web*. Journal of Chemometrics, 38 (7), 2024. DOI: [10.1002/cem.3556](http://dx.doi.org/10.1002/cem.3556).

2. A. Pomerantsev, O. Rodionova. *Selectivity in Nontargeted Qualitative Analysis*. Analytica Chimica Acta, 1332, 2024. DOI: [10.1016/j.aca.2024.343352](https://doi.org/10.1016/j.aca.2024.343352).

3. A. Pomerantsev, O. Rodionova. *Popular decision rules in SIMCA: Critical review*. Journal of Chemometrics, 34 (8), 2020. DOI: [10.1002/cem.3250](https://doi.org/10.1002/cem.3250)

4. A. Pomerantsev, O. Rodionova. *On the type II error in SIMCA method*. Journal of Chemometrics, 28 (6), 2014. DOI: [10.1002/cem.2610](https://doi.org/10.1002/cem.2610).
