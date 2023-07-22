# medMNIST NCA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains an implementation of NCA (Neural Cellular Automata) for the medMNIST dataset. The medMNIST dataset is a collection of medical image classification datasets in the MNIST format. NCA is a generative model that can be used to generate images.

## Inspiration
This project was inspired by the [Growing Neural Cellular Automata paper](https://doi.org/10.23915/distill.00023) and done as a project for the lecture Deep Generative Methods at the TU Darmstadt.

The original code of the paper can be found [here](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb) and was adapted to our application.

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage

To run the NCA algorithm on the medMNIST dataset use the jupyter notebook `medmnist_nca.ipynb`. The notebook is originally desinged to run on Kaggle, but can be run locally as well. To run the notebook locally, download the medMNIST dataset from [here](https://medmnist.com/) and place it in the same folder. Then run the notebook.


