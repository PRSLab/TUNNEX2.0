TUNNEX2.0
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/PRSLab/TUNNEX2.0/workflows/CI/badge.svg)](https://github.com/PRSLab/TUNNEX2.0/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/PRSLab/TUNNEX2.0/branch/main/graph/badge.svg)](https://codecov.io/gh/PRSLab/TUNNEX2.0/branch/main)

Python version of TUNNEX for computing WKB tunneling rates.

## Installation

```bash
git clone https://github.com/PRSLab/TUNNEX2.0.git
cd TUNNEX2.0
conda create -n tunnex_2 python=3.10
conda activate tunnex_2
pip install -e .
```

## Usage

```bash
python -m tunnex_2 [options] files ...
python -m tunnex_2 --help
```

See the `docs/` directory for the full documentation
(build it with `sphinx-build -b html docs docs/_build/html`).

### Copyright

Copyright (c) 2026, Pavel Zasimov

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.