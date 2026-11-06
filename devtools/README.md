# Development and testing tools

This directory contains helper files for testing the project and setting up conda environments.

## Contents

* `conda-envs/` - YAML files describing conda environments
  * `environment.yml` - snapshot of the development environment (`tunnex_2`), exported with `conda env export --from-history`
  * `test_env.yml` - environment used for testing (Python, pytest, pytest-cov, codecov, and the project dependencies)
* `scripts/`
  * `create_conda_env.py` - creates a conda environment from a YAML file with a given name and Python version

## Creating the test environment

The script requires PyYAML (`conda install pyyaml`).

```bash
python devtools/scripts/create_conda_env.py -n test -p 3.13 devtools/conda-envs/test_env.yml
conda activate test
pytest
```

The script replaces the `python` entry in the YAML file with the requested version and calls `conda env create`.
The original YAML file is not modified.

## Updating the environment snapshot

```bash
conda activate tunnex_2
conda env export --from-history -f devtools/conda-envs/environment.yml
```

Use the `-f` flag instead of `>` redirection: in Windows PowerShell, `>` writes UTF-16, which PyYAML cannot read.
Packages installed via `pip` are not included in `--from-history` output; add them to a `pip:` block manually.

## Versioning

The package version is determined automatically by [Versioneer](https://github.com/warner/python-versioneer)
from git tags. To set a version, create a tag:

```bash
git tag -a 0.1.0 -m "first version"
```