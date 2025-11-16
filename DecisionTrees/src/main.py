from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeRegressor import DecisionTreeRegressor 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor

from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def test_classifier():
    """Tests the DecisionTreeClassifier on the Iris dataset and compares with scikit-learn."""
    print("="*55)
    print("Testing Custom Classifier vs. Scikit-learn")
    print("="*55)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    your_clf = DecisionTreeClassifier(max_depth=5)
    your_clf.fit(X_train, y_train)
    your_predictions = your_clf.predict(X_test)
    your_accuracy = accuracy_score(y_test, your_predictions)

    sklearn_clf = SklearnDTClassifier(max_depth=5, random_state=42)
    sklearn_clf.fit(X_train, y_train)
    sklearn_predictions = sklearn_clf.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    
    print(f"Dataset: Iris")
    print(f"Max Depth: 5")
    print("-" * 30)
    print(f"Custom Classifier Accuracy:   {your_accuracy * 100:.2f}%")
    print(f"Scikit-learn Accuracy: {sklearn_accuracy * 100:.2f}%")
    print("\n")


def test_regressor():
    """Tests the DecisionTreeRegressor and compares with scikit-learn."""
    print("="*55)
    print("Testing Custom Regressor vs. Scikit-learn")
    print("="*55)
    
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(200, 1), axis=0)
    y = np.sin(X).ravel() + (0.2 * np.random.randn(200))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    your_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    your_reg.fit(X_train, y_train)
    your_predictions = your_reg.predict(X_test)
    your_rmse = np.sqrt(mean_squared_error(y_test, your_predictions))
    
    sklearn_reg = SklearnDTRegressor(max_depth=5, random_state=42)
    sklearn_reg.fit(X_train, y_train)
    sklearn_predictions = sklearn_reg.predict(X_test)
    sklearn_rmse = np.sqrt(mean_squared_error(y_test, sklearn_predictions))
    
    print(f"Dataset: Synthetic Sine Wave")
    print(f"Max Depth: 5")
    print("-" * 30)
    print(f"Custom Regressor RMSE:   {your_rmse:.4f}")
    print(f"Scikit-learn RMSE: {sklearn_rmse:.4f}")
    print("\n")


if __name__ == "__main__":
    test_classifier()
    test_regressor()
