# Teeth Caries Classifier

This project is a deep learning-based application that classifies images of teeth into two categories: **caries** (tooth decay) and **without caries**. The classifier is implemented using Convolutional Neural Networks (CNNs) and trained on a dataset of teeth images. The project also includes data augmentation, visualization, and performance evaluation.

## Table of Contents

1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Installation and Usage](#installation-and-usage)
4. [Results and Visualizations](#results-and-visualizations)
5. [Files and Directory Structure](#files-and-directory-structure)
6. [Future Improvements](#future-improvements)

## Dataset

The dataset is structured into the following directories:

```
teeth_dataset/
├── Trianing/
│   ├── caries/
│   └── without_caries/
├── test/
    ├── caries/
    └── no-caries/
```

- **Training Data**: Contains subfolders for `caries` and `without_caries` classes.
- **Test Data**: Separate subfolders for `caries` and `no-caries`.

Images are resized to 128x128 pixels before being fed into the model.

## Model Architecture

The CNN model has the following layers:

1. **Convolutional Layers**: Extract features using 3x3 kernels.
2. **MaxPooling Layers**: Reduce spatial dimensions.
3. **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
4. **Dense Layers**: Fully connected layers with ReLU activation.
5. **Dropout Layer**: Prevents overfitting by randomly dropping connections.
6. **Output Layer**: Single neuron with sigmoid activation for binary classification.

The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

## Installation and Usage

### Prerequisites

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yashkk07/CariesDetection
   cd CariesDetection
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `teeth_dataset` directory, structured as shown above.

4. Run the training script and evaluate the model on the test set:
   ```bash
   python teeth.py
   ```

## Results and Visualizations

### Accuracy and Loss Plots

The training and validation accuracy and loss are plotted and saved in the `plots` directory:

- `loss_plot.png`

### Class Distribution

A bar graph displaying the number of images in each category is saved as `class_distribution.png`.

### Example Classes

Sample images from the test set with their labels are displayed.

### Prediction Probabilities

The distribution of prediction probabilities is visualized in `probability_histogram.png`.

### Confusion Matrix

The confusion matrix summarizes the model's performance by showing true positives, true negatives, false positives, and false negatives. It helps evaluate the model's accuracy and balance.  
Saved as `confusion_matrix.png`.

## Files and Directory Structure

```
project_root/
├── train_classifier.py        # Training script
├── evaluate_model.py          # Evaluation script
├── teeth_dataset/             # Dataset and plots
│   ├── Trianing/              # Training data
│   ├── test/                  # Test data
│   ├── plots/                 # Saved plots
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

## Future Improvements

1. **Expand Dataset**: Add more diverse samples to improve generalization.
2. **Hyperparameter Tuning**: Experiment with different optimizers, learning rates, and architectures.
3. **Explainability**: Implement Grad-CAM or similar techniques to visualize decision-making.
4. **Deployment**: Create a web-based or mobile application for real-world use.

---

Feel free to contribute by submitting issues or pull requests!
