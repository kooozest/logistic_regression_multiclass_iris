# Multiclass Logistic Regression with the Iris Dataset 🌸

![Iris Dataset](https://img.shields.io/badge/Iris%20Dataset-Logistic%20Regression-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

Welcome to the **logistic_regression_multiclass_iris** repository! This project provides a clear and educational implementation of multiclass logistic regression using the classic Iris flower dataset. You can find the code and documentation [here](https://github.com/kooozest/logistic_regression_multiclass_iris/releases). 

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Dataset Description](#dataset-description)
- [Implementation Details](#implementation-details)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to demonstrate the application of logistic regression in a multiclass classification scenario. The Iris dataset contains three classes of iris flowers, each represented by four features: sepal length, sepal width, petal length, and petal width. 

By the end of this project, you will have a solid understanding of how logistic regression works for multiclass problems, along with practical experience using Python libraries such as Pandas and Scikit-learn.

## Getting Started

To get started, you will need to clone this repository to your local machine. Use the following command:

```bash
git clone https://github.com/kooozest/logistic_regression_multiclass_iris.git
```

### Prerequisites

Ensure you have Python 3.8 or higher installed. You will also need to install the required libraries. You can do this using pip:

```bash
pip install pandas scikit-learn matplotlib
```

### Running the Code

Once you have cloned the repository and installed the necessary libraries, navigate to the project directory and run the main script:

```bash
python logistic_regression.py
```

You can download the latest release [here](https://github.com/kooozest/logistic_regression_multiclass_iris/releases) and execute the file to see the results.

## Dataset Description

The Iris dataset consists of 150 samples, with each sample belonging to one of three species of iris flowers: Setosa, Versicolor, and Virginica. Each sample is described by four features:

- **Sepal Length**: Length of the sepal in centimeters.
- **Sepal Width**: Width of the sepal in centimeters.
- **Petal Length**: Length of the petal in centimeters.
- **Petal Width**: Width of the petal in centimeters.

### Data Visualization

To better understand the dataset, we visualize it using scatter plots. This helps in identifying patterns and relationships between the features.

![Iris Scatter Plot](https://upload.wikimedia.org/wikipedia/commons/5/5c/Iris_flower_scatter_plot.png)

## Implementation Details

### Logistic Regression

Logistic regression is a statistical method for predicting binary classes. However, it can be extended to handle multiclass problems through techniques such as one-vs-rest (OvR). In this project, we will implement the OvR approach.

### Steps Involved

1. **Data Loading**: Load the Iris dataset using Pandas.
2. **Data Preprocessing**: Clean the data and prepare it for modeling.
3. **Model Training**: Train the logistic regression model using Scikit-learn.
4. **Model Evaluation**: Evaluate the model's performance using accuracy and confusion matrix.

### Code Snippet

Here’s a brief look at how we load the dataset and prepare it for training:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('iris.csv')

# Split the data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Code Structure

The repository is organized as follows:

```
logistic_regression_multiclass_iris/
│
├── data/
│   └── iris.csv                # Iris dataset
│
├── src/
│   ├── logistic_regression.py  # Main script for logistic regression
│   └── utils.py                # Utility functions
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for data analysis
│
├── requirements.txt            # Required packages
└── README.md                   # Project documentation
```

### Main Script

The main script, `logistic_regression.py`, contains the core logic for training and evaluating the model. 

### Utility Functions

The `utils.py` file contains helper functions for data loading, preprocessing, and visualization.

## Results

After running the model, you will obtain accuracy metrics and a confusion matrix. This helps in assessing how well the model performs on the test data.

### Example Output

```plaintext
Accuracy: 95.0%
Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add features, feel free to fork the repository and submit a pull request. 

### How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For the latest updates and releases, visit the [Releases](https://github.com/kooozest/logistic_regression_multiclass_iris/releases) section.

Feel free to explore, learn, and enhance your understanding of logistic regression with this educational project!