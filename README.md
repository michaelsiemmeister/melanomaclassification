# README

This repository contains the source code for `melanomaclassification`.

The goal of this project is to provide an easy way to train and analyse
neural nets for melanoma-naevus classification. 

This project uses images from <www.isic-archive.com>. For analysis and
comparison to existing models, the ISIC Challenge 2016 and MClass are used. 

### References:

for the ISIC Challenge 2016 dataset:

Brinker, T.J., Hekler, A., Hauschild, A., Berking, C., Schilling, B., Enk,
A.H., Haferkamp, S., Karoglan, A., von Kalle, C., Weichenthal, M. and
Sattler, E., 2019. Comparing artificial intelligence algorithms to 157 German
dermatologists: the melanoma classification benchmark. European Journal of
Cancer, 111, pp.30-37.

For the MClass dataset:

Gutman, D., Codella, N.C., Celebi, E., Helba, B., Marchetti, M., Mishra, N.
and Halpern, A., 2016. Skin lesion analysis toward melanoma detection: A
challenge at the international symposium on biomedical imaging (ISBI) 2016,
hosted by the international skin imaging collaboration (ISIC). arXiv preprint
arXiv:1605.01397.

### Project structure

The project is divided into tasks.

1. Download images

2. Setup pandas dataframes for various datasets

3. Train and Save the model

4. Load and Predict

This project has been tested on GNU/Linux Ubuntu 18.04LTS. 
The folder `.vscode` contains Visual Studio Code settings. It can be deleted
without any serious consequences.



# Installation


## Very important: pipenv version

Install a new version of pipenv from Github. As of 2019-11-12 the standard
version of pipenv from pip does not work. I installed a version from Github.
The installation does work with version `pipenv, version 2018.11.27.dev0`.

To install pipenv 

```shell
pyenv shell TheVersionYouWant 
git clone https://github.com/pypa/pipenv.git
cd pipenv
pip install -e .
```

## Standard installation procedure. 

- `git clone` this repository.

- make sure that you have `pyenv` installed.

- `pyenv install 3.7.4`

- `pyenv shell 3.7.4`

- `pip install pipenv`

- `pipenv install --dev` (`--dev`) is optional

- `cp example.config.yaml config.yaml` is optional. When started from Jupyter Lab notebooks, it `melanomclassification.py` only reads `example.config.yaml`. See `melanomaclassification.py`.

- modify `config.yaml` as you like

- `cp .env.example .env`

- `pipenv shell` to activate the virtual environment

- in the virtual environment: 
  `python melanoma_classification --config_file_path=config.yaml`



# Testing. 


`py.test` is used for testing. To run the tests use 
`pipenv run python -m pytest`. 

Bad news: `pytest` inside `pipenv shell` does not work.

Adding `-s` like in 

```
pipenv run python -m pytest melanoma_classification/lib/test_categories.py -s
```

disables stdout and stderr capturing.

## Specific Tests.

To run a specific test function use a command like

```
pipenv run python -m pytest 
melanoma_classification/lib/test_categories.py::test_create_test_df -s
```

where the function name follows the `::`.

# Running some modules.

The environment variable PYTHONPATH has to be set correctly. Use a `.env` file. Just
rename the `.env.example` to `.env`. Then use `pipenv shell` to load an
environment with the correct path set.

# Running Jupyter Lab.

```shell
pyenv shell 3.7.4 # activate correct python version
pipenv shell
jupyter lab
```

