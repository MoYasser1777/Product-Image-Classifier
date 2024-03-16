from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def train():
    # Initialize lists to store images and labels
    images = []
    labels = []

    # Get all the image folder paths
    image_folders = os.listdir("./Images")

    for folder in image_folders:
        # Get all the image names
        all_images = os.listdir(f"./Images/{folder}")

        # Iterate over the image names, get the label
        for img_name in all_images:
            img_path = f"./Images/{folder}/{img_name}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))

            # Get the HOG descriptor for the image
            hog_desc = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # Update the data and labels
            images.append(hog_desc)
            labels.append(folder)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the parameter grid to search
    param_grid = {'C': [0.1, 1, 10, 100]}

    # Initialize the grid search cross-validation
    grid_search = GridSearchCV(LinearSVC(random_state=42, tol=1e-5), param_grid, cv=5)

    # Perform grid search cross-validation
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by grid search
    print("Best Parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Validate the model
    val_accuracy = best_model.score(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy}')

    # Save the best model
    joblib.dump(best_model, "HOG_Model_DIGITS_best.npy")


def get_prediction(image):
    HOG_model = joblib.load("HOG_Model_DIGITS.npy")

    resized_img = cv2.resize(image, (28, 28))
    # Get the HOG descriptor for the test image
    hog_desc = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    # Prediction
    pred = HOG_model.predict(hog_desc.reshape(1, -1))[0]

    # Return the predicted label
    return pred.title()
