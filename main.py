import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def regression_analysis(x, y):
    # Convert data to pandas DataFrame
    data = pd.DataFrame({'X': x, 'Y': y})
    
    # Step 1: Display the data
    print("Step 1: Display the data")
    print(data)
    print()

    # Step 2: Summary statistics
    print("Step 2: Summary statistics")
    print(data.describe())
    print()

    # Step 3: Fit the regression model
    X = sm.add_constant(data['X'])  # Adds a constant term to the predictor
    model = sm.OLS(data['Y'], X).fit()
    
    # Step 4: Display regression coefficients
    print("Step 3: Regression equation")
    print("Y = {:.4f} + {:.4f}X".format(model.params['const'], model.params['X']))
    print()

    # Plotting the regression line
    plt.scatter(x, y, label='Data points')
    plt.plot(data['X'], model.fittedvalues, color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Regression Analysis')
    plt.show()

# Example usage
try:
    x = list(map(float, input("Enter the independent variable data (space-separated values): ").split()))
    y = list(map(float, input("Enter the dependent variable data (space-separated values): ").split()))
    
    if len(x) != len(y):
        raise ValueError("The number of X and Y values must be the same.")
    
    regression_analysis(x, y)
except ValueError as e:
    print(f"Invalid input: {e}")
