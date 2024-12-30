# EUR/USD Price Prediction

This repository contains code for predicting EUR/USD prices using a hybrid CNN-LSTM model. The code includes data preprocessing, technical indicator calculations, model training, and evaluation.

## Repository Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for data preprocessing and exploration.
- `src/`: Source code including technical indicators, model definition, and evaluation.
- `saved_models/`: Directory to save trained models.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.
- `requirements.txt`: List of dependencies required to run the code.
- `LICENSE`: License information.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/eurusd-prediction.git
    cd eurusd-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Place the dataset (`eurusd_15min_data.csv`) in the `data/` directory.

## Usage

1. Run the data preprocessing notebook:
    ```sh
    jupyter notebook notebooks/data_preprocessing.ipynb
    ```

2. Run the training script:
    ```sh
    python src/model.py
    ```

3. Evaluate the model:
    ```sh
    python src/evaluation.py
    ```

## Model Architecture

The model is a hybrid CNN-LSTM network designed to capture both spatial and temporal dependencies in the data. It consists of:
- Convolutional layers for feature extraction
- LSTM layers for sequence modeling
- Dense layers for output prediction

## Results

The model's performance metrics and visualizations of actual vs predicted prices are provided in the evaluation script.

## Acknowledgements

- TensorFlow and Keras for providing the tools to build and train the model.
- Pandas and NumPy for data manipulation.
- Matplotlib for visualization.
