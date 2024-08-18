# Leaf-Disease-Classification-using-Pretrained-ResNet-50

This repository contains code for classifying leaf diseases using a pretrained ResNet-50 model. The project uses transfer learning techniques to leverage the powerful ResNet-50 architecture, which is trained on the ImageNet dataset, to accurately classify leaf diseases from a dataset of plant images.

## Project Overview

The primary goal of this project is to build a robust model that can identify various plant diseases from images of leaves. By using a pretrained ResNet-50 model, we can achieve high accuracy with limited computational resources and training time. The project uses the "New Plant Diseases Dataset" from Kaggle, which contains images of leaves labeled with various diseases.

## Dataset

The dataset used in this project is the "New Plant Diseases Dataset" from Kaggle. The dataset includes images of healthy and diseased leaves across multiple plant species.

- **Number of Classes:** 38

## Project Structure

- **`leaf_disease_classification.ipynb`:** The main Colab notebook containing all the code required to train and evaluate the model.
- **`kaggle.json`:** JSON file containing the Kaggle API credentials to download the dataset.
- **`model.keras`:** Saved model file of the best-performing model during training.

## Dependencies

Before running the code, ensure that you have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- PIL (Pillow)
- Kaggle API

You can install the required libraries using pip:

```bash
pip install -q tensorflow keras numpy pandas matplotlib opencv-python pillow kaggle
```

## Usage

To run the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/leaf-disease-classification-resnet50.git
   cd leaf-disease-classification-resnet50
   ```

2. **Set Up Kaggle API:**
   Upload your `kaggle.json` file to the Colab environment:
   ```python
   from google.colab import files
   files.upload()
   ```

   Move the `kaggle.json` file to the correct location:
   ```bash
   ! mkdir ~/.kaggle
   ! cp kaggle.json ~/.kaggle/
   ! chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download and Unzip the Dataset:**
   ```bash
   !kaggle datasets download -d vipoooool/new-plant-diseases-dataset
   !unzip new-plant-diseases-dataset.zip
   ```

4. **Train the Model:**
   Run the cells in the `leaf_disease_classification.ipynb` notebook to train the model on the dataset.

5. **Evaluate the Model:**
   The notebook includes code to evaluate the model's performance on the validation set.

## Model Architecture

The model uses the ResNet-50 architecture with the following modifications:

- **Input Layer:** 224x224x3 image input.
- **Base Model:** Pretrained ResNet-50 with ImageNet weights (without the top layer).
- **Global Average Pooling:** Reduces the feature maps to a single vector.
- **Fully Connected Layers:** Two Dense layers with ReLU activation and a final output layer with softmax activation for classification.

## Image Augmentation

The training data is augmented using the `ImageDataGenerator` class in Keras, which applies transformations such as:

- Shear
- Zoom
- Horizontal and Vertical Shifts
- Filling mode for missing pixels

## Callbacks

The training process uses the following callbacks:

- **EarlyStopping:** Stops training if the validation accuracy does not improve for 7 epochs.
- **ModelCheckpoint:** Saves the best model based on validation accuracy.
- **ReduceLROnPlateau:** Reduces the learning rate if the validation accuracy plateaus.

## Results

After training for 5 epochs, the model achieves an accuracy of 97% on the validation set.
