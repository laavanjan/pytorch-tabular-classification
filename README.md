# Tabular Data Classification with PyTorch

This project demonstrates how to build a binary classification model for tabular data using PyTorch. The workflow is implemented in the notebook `Tabular_Data_Classification.ipynb` and uses a rice type classification dataset from Kaggle.

## Features
- **Dataset Download:** Automatically downloads the rice type classification dataset using `opendatasets`.
- **Data Preprocessing:** Cleans the data by removing missing values and normalizes features for better model performance.
- **Data Splitting:** Splits the data into training, validation, and test sets.
- **PyTorch Dataset & DataLoader:** Wraps the data in custom PyTorch Dataset objects and uses DataLoader for efficient batching.
- **Model Definition:** Implements a simple feedforward neural network for binary classification.
- **Training & Validation:** Trains the model and tracks loss and accuracy for both training and validation sets.
- **Testing:** Evaluates the model on the test set and reports accuracy.
- **Visualization:** Plots training/validation loss and accuracy over epochs.
- **Inference:** Allows user input for feature values to make predictions with the trained model.

## Usage
1. Open `Tabular_Data_Classification.ipynb` in Jupyter or VS Code.
2. Run all cells sequentially to:
   - Download and preprocess the dataset
   - Train and evaluate the model
   - Visualize results
   - Make predictions using custom input

## Requirements
- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- pandas
- numpy
- opendatasets
- torchsummary

Install dependencies with:
```python
!pip install opendatasets torch torchsummary scikit-learn matplotlib pandas numpy
```

## Dataset
- [Rice Type Classification Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)

## Notes
- The notebook is designed for educational purposes and can be adapted for other tabular classification tasks.
- GPU acceleration is supported if available.

## License
This project is provided for educational use. Please check the dataset license before using for commercial purposes.
