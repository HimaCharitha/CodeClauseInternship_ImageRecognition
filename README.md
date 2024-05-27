# CodeClauseInternship_Image_Recognition
# Image Classifier using CNN

## Aim
The aim of this project is to build an image classifier that can classify images into predefined categories using a trained Convolutional Neural Network (CNN) model.

## Description
This program uses a CNN to classify images from the CIFAR-10 dataset into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The model is built using TensorFlow and Keras.

## Technologies
- Python
- TensorFlow
- Keras
- CIFAR-10 Dataset

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/image-classifier-cnn.git
    cd image-classifier-cnn
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the script to train the model:
    ```sh
    python train.py
    ```

2. Evaluate the model on the test set:
    ```sh
    python evaluate.py
    ```

3. Use the model to make predictions on new images:
    ```sh
    python predict.py --image_path path/to/image.jpg
    ```

## Project Structure
image-classifier-cnn/
│
├── data/
│ └── cifar-10/ # Directory for the CIFAR-10 dataset
│
├── models/
│ └── cnn_model.h5 # Trained model
│
├── notebooks/
│ └── cnn_model.ipynb # Jupyter notebook for model development
│
├── src/
│ ├── train.py # Script to train the model
│ ├── evaluate.py # Script to evaluate the model
│ └── predict.py # Script to make predictions using the model
│
├── README.md # This file
├── requirements.txt # List of dependencies
└── .gitignore # Git ignore file

## Example
To train the model, run:
```sh
python src/train.py
To evaluate the model, run:
python src/evaluate.py
To make a prediction on a new image, run:
python src/predict.py --image_path path/to/image.jpg
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

### `requirements.txt`
Add a `requirements.txt` file for the necessary Python packages:
