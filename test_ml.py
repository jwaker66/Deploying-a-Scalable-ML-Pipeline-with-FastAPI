import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Ensure the `ml.model` module is importable
root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

from ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    """
    Test pipeline of training model
    """
    # Create dummy data
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    
    # Train the model
    model = train_model(X, y)
    
    # Check if the model is an instance of BaseEstimator and ClassifierMixin
    assert isinstance(model, BaseEstimator) and isinstance(model, ClassifierMixin)

def test_compute_model_metrics():
    """
    Test compute_model_metrics
    """
    # Dummy true and predicted labels
    y_true = np.array([1, 1, 0])
    y_preds = np.array([0, 1, 1])
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    
    # Check if metrics are not None
    assert precision is not None
    assert recall is not None
    assert fbeta is not None

def test_inference():
    """
    Test inference of model
    """
    # Create dummy data
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    
    # Train the model
    model = train_model(X, y)
    
    # Perform inference
    y_preds = inference(model, X)
    
    # Check if predictions have the same shape as the input
    assert y_preds.shape == y.shape