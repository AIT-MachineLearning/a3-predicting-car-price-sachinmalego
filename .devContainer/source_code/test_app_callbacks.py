import numpy as np
from pages.model_vtwo import get_X, predict_model

# Define feature values for testing
feature_vals = [2017, 70000, 24.3, 1248]

# Define labels for expected outputs
labels = ['Cheap', 'Affordable', 'Expensive', 'Very Expensive']

# Define possible expected outputs
possible_outputs = [label for label in labels]

# Test the get_X
def test_get_X():
    # Get feature values and features
    #output = get_X(*feature_vals)
    #assert output == (2017,70000,24.3,1248), f'Output: {output}'

    # Get feature values and features
    X, features = get_X(*feature_vals)
    
    # Assert the shape and data type of X
    assert X.shape == (1, 4)
    assert X.dtype == np.float64

    # Get predicted y values
    #y = get_y(X)

    # Assert the shape of y
    #assert y.shape == (1,)


# Test the predict_selling_price callback function
def test_selling_price():
    output = predict_model(*feature_vals)
    #assert output[0] in possible_outputs
    if isinstance(output, str):
        output = np.array([output])
        
    assert output.shape == (1,), f'Expected output shape: (1,), Output shape received: {output.shape}'