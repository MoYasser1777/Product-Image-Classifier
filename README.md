# Product-Image-Classifier

This repository contains code for an image classification system using a Support Vector Machine (SVM) algorithm trained on Histogram of Oriented Gradients (HOG) features. The system is capable of classifying images into three main classes: fashion, nutrition, and accessories.

Overview:

- Training: The train() function preprocesses images from three main classes (fashion, nutrition, and accessories) by extracting HOG features. It then trains an SVM classifier using grid search cross-validation to find the optimal hyperparameters. The best model is selected based on the validation accuracy, achieving an accuracy of 79% on the validation set.

- Prediction: The get_prediction(image) function loads the trained SVM model and takes an input image. It preprocesses the image by resizing it and extracting HOG features. Then, it predicts the class label for the image using the trained model.

Requirements:
Python 3.x
scikit-learn
scikit-image
joblib

Usage:
1- Training: Place your image datasets into the Images folder, organizing them into subfolders representing different classes. Run the train() function to train the SVM classifier.

2- Prediction: After training, you can use the get_prediction(image) function to predict the class of a new image. Provide the path to the image as input.

Note:
- Ensure that your image datasets are well-organized in the Images folder, with each subfolder representing a different class.
- Fine-tuning the model architecture or hyperparameters might further improve classification accuracy.
