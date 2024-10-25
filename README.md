# ML-Lab
# Linear Regression Assignment - Nairobi Office Price Prediction
## Machine Learning Lab

### Overview
This Python script implements a linear regression model to predict office prices in Nairobi based on office size. The assignment demonstrates the use of gradient descent and Mean Squared Error (MSE) for a simple predictive model.

### Requirements
- Python 3.x
- Required libraries: numpy, pandas, matplotlib
```bash
pip install numpy pandas matplotlib
```

### Dataset
Uses `NairobiOfficePriceEx.csv` containing office data with the following relevant columns:
- SIZE: Office size in square feet
- PRICE: Office price in thousands of dollars

### How to Run
1. Make sure `NairobiOfficePriceEx.csv` is in the same folder as your Python script
2. Run the script:

```
python predictive_model.py
```

### What the Code Does
1. Loads and preprocesses the data
2. Implements two main functions:
   - Mean Squared Error calculation
   - Gradient Descent algorithm
3. Trains for 10 epochs
4. Shows training progress and final results
5. Creates visualizations

### Output
The program will show:
1. Training progress with MSE for each epoch
2. Final model parameters
3. Price prediction for 100 sq ft office
4. Two plots:
   - Data points with regression line
   - Error over training epochs

### Assignment Components Covered
- [x] Data loading and preprocessing
- [x] MSE implementation
- [x] Gradient Descent implementation
- [x] Model training
- [x] Visualization
- [x] Price prediction

### Author
[Maiyo Dennis]
[Student ID - 151735]