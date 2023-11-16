# Telecom Customer Churn Prediction

## Overview

This project aims to predict customer churn in a telecommunications company using machine learning. The dataset used for training and evaluation contains various features related to customer interactions and characteristics.

## Project Structure

- **Data Exploration:**
  - Explored the dataset to understand its structure and characteristics.
  - Checked for missing values and handled them appropriately.
  - Visualized the distribution of numerical and categorical features.

- **Data Preprocessing:**
  - Dropped irrelevant columns (e.g., customerID).
  - Converted the 'TotalCharges' column to numeric, handling missing values.
  - Encoded non-numeric variables using Label Encoding.
  - Checked and handled correlations between features.

- **Feature Selection:**
  - Utilized correlation analysis to select relevant features for modeling.

- **Modeling:**
  - Implemented a simple neural network using Keras for initial evaluation.
  - Applied GridSearchCV to find the best hyperparameters for the neural network model.

- **Evaluation:**
  - Assessed model performance before and after hyperparameter tuning.
  - Utilized metrics such as accuracy and AUC score for evaluation.

- **Model Deployment:**
  - Saved the best-performing model and the associated scaler for future use.
  - I have attached a video showing how the models perform and the code for the deployment

## Files

- **`best_model.pkl`:** Saved model file containing the best-performing neural network.
- **`scaler.pkl`:** Saved scaler file used for feature scaling.
- The Jupyter Notebook files are also available for your use.

## Dependencies

- Python 3.x
- Libraries: Pandas, NumPy, Seaborn, TensorFlow, Keras, Scikit-Learn, Matplotlib.

## Usage

1. Ensure the required dependencies are installed (use `pip install -r requirements.txt`).
2. Run the Jupyter notebook (`Telecom_Churn_Prediction.ipynb`) to train and evaluate the model.
3. The best model and scaler will be saved for future use.

