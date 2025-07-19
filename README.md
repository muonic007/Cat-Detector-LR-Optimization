# Cat vs Non-Cat Classifier with Logistic Regression and Learning Rate Optimization

## Project Description  
This project implements a logistic regression model to classify images as cat or non-cat, using 64x64 color images for training.  
The focus is on optimizing the learning rate to find the best value that yields the highest accuracy on the test dataset.

## Features  
- Load and preprocess Cat vs Non-Cat dataset from HDF5 files  
- Implement logistic regression model including cost function and gradient descent  
- Train the model with adjustable learning rate  
- Evaluate model accuracy across different learning rates  
- Plot Accuracy vs. Learning Rate to select the optimal learning rate

## Code Structure  
- `load_dataset()`: loads the data  
- `initialize()`, `sigmoid()`, `propagate()`, `optimize()`, `predict()`, `model()`: implement the logistic regression model  
- Main script prepares data, trains model with various learning rates, and records accuracy  
- Generates a plot showing accuracy as a function of learning rate

## How to Run  
1. Ensure the files `train_catvnoncat.h5` and `test_catvnoncat.h5` are in the project folder.  
2. Install required packages: `numpy`, `h5py`, and `matplotlib`.  
3. Run the code; the model trains with multiple learning rates and accuracy for each is computed.  
4. A plot of accuracy vs learning rate is displayed to help select the best learning rate.

## Results  
- The model achieves decent accuracy in classifying cat images.  
- The plot demonstrates the sensitivity of accuracy to the learning rate and shows how selecting the right learning rate directly impacts model performance.

## Requirements  
- Python 3.x  
- numpy  
- h5py  
- matplotlib  

## Future Improvements  
- Implement advanced optimizers like Adam or adaptive learning rates.  
- Incorporate deeper or more complex models for better accuracy.

