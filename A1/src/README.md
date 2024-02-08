# Project no. A1 “activity recognition with four accelerometers”

## Overview
Recogize human activities with four accelerometes.

## Reference papers
[Fadel19] Fadel, W. F., Urbanek, J. K., Albertson, S. R., Li, X., Chomistek, A. K., & Harezlak, J., Differentiating Between Walking and Stair Climbing Using Raw Accelerometry Data, in Statistics in Biosciences, 11(2), 334– 354, 2019.

[Karas19] Karas, M., Bai, J., Strączkiewicz, M., Harezlak, J., Glynn, N. W., Harris, T., ... Urbanek, J. K., Accelerometry Data in Health Research: Challenges and Opportunities. Review and Examples, in Statistics in Biosciences, 11, 210–23, 2019.

## Dataset
Dataset (760.8 MB uncompressed)
https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/

## How to Use

Follow the steps below to install PyEnv, create and activate the virtual environment for this project:

### Prerequisites

Make sure you have the following prerequisites installed on your system:
- Python (recommended version: 3.8.10)
- PyEnv (https://github.com/pyenv/pyenv)

Alternatively, you can use the following commands to install the required dependencies:

```shell
pip3 install virtualenv
pip3 install virtualenvwrapper
brew install pyenv-virtualenv
```

Next, install Python 3.8.10 using PyEnv:

```shell
pyenv install 3.8.10
```

Create a virtual environment named 'HDA' for this project:

```shell
pyenv virtualenv 3.8.10 HDA
```

Activate the virtual environment in your working directory:

```shell
pyenv activate HDA
```

Install the required packages using pip:

```shell
pip install pystan==2.19.1.1
pip install wget
```

### Installation

1. Clone this repository:

```shell
git clone https://github.com/alvindotai/time-series-analysis.git
```

2. Navigate to the project directory:

```shell
cd A1
```

#### Project Structure
The project follows the following structure:

```
A1/
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── external/
  │
  ├── notebooks/
  │   ├── EDA.ipynb
  │
  ├── src/
  │   ├── data/
  │   │   └── data_preprocessing.py
  │   ├── logs/
  │   ├── models/
  │   │   ├── DeepConvLSTM.py
  │   │   └── DNN.py
  │   │   └── LSTM.py
  │   │   └── CNN.py
  │   ├── trained_models/
  │   ├── plots/
  │   ├── utils/
  │   └── main.py
  │
  ├── config/
  │
  ├── requirements.txt
  ├── README.md

```


