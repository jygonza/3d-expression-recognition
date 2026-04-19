# Random Forest 3D Facial Expression Recognition

## Overview
This project classifies facial expressions from 3D facial landmark data stored in `.bnd` files. Each file contains 83 facial landmarks, where each landmark has `(x, y, z)` coordinates. The project supports five input data types:

- `o` - original landmarks
- `t` - translated landmarks (centered at the origin)
- `x` - landmarks rotated 180 degrees around the x-axis
- `y` - landmarks rotated 180 degrees around the y-axis
- `z` - landmarks rotated 180 degrees around the z-axis

The script builds feature vectors from the landmark coordinates and evaluates a Random Forest classifier using Leave-One-Subject-Out (LOSO) cross-validation.

## Main file
- `main.py` - loads the data, applies the selected transformation, runs LOSO cross-validation, prints evaluation metrics, and saves results to a text file.

## Imported packages used in `main.py`

### Python standard library
- `os` - walks through directories, builds file paths, checks whether output directories exist, and creates the results directory.
- `sys` - reads command-line arguments passed to the script.
- `time` - measures the total runtime of the program.
- `dataclasses.dataclass` - defines the `Sample` dataclass used to store one example's features, label, subject group, and file path.
- `math.acos`, `math.cos`, `math.sin` - computes pi and rotation matrices for the 180 degree landmark rotations.
- `typing.List`, `typing.Tuple` - provides type hints for function parameters and return values.

### Third-party packages
- `numpy` - stores landmark coordinates as arrays, computes means for translation, performs matrix multiplication for rotations, and builds feature matrices for training.
- `sklearn.ensemble.RandomForestClassifier` - the classifier used to train and predict facial expression labels.
- `sklearn.model_selection.LeaveOneGroupOut` - performs Leave-One-Subject-Out cross-validation using subject IDs as groups.
- `sklearn.metrics.confusion_matrix` - computes the confusion matrix.
- `sklearn.metrics.accuracy_score` - computes classification accuracy.
- `sklearn.metrics.f1_score` - computes macro F1 score.
- `sklearn.metrics.precision_score` - computes macro precision.
- `sklearn.metrics.recall_score` - computes macro recall.

## Installation
Create and activate your Python environment, then install the required third-party packages:

```bash
pip install numpy scikit-learn
```

## How to run the project
From the command line:

```bash
python Project1.py <data_type> <data_directory>
```

### Arguments
- `<data_type>` must be one of: `o`, `t`, `x`, `y`, `z`
- `<data_directory>` should be the root folder containing the subject folders for the dataset

### Example
```bash
python main.py o ./FacialLandmarks/BU4DFE_BND_V1.1
```

## Expected dataset structure
The script assumes a nested folder structure like this:

```text
BU4DFE_BND_V1.1/
  M001/
    Angry/
      sample1.bnd
      sample2.bnd
    Happy/
      sample1.bnd
  M002/
    Sad/
      sample1.bnd
```

Each `.bnd` file is read into an `83 x 3` NumPy array. The code then transforms the data, flattens it when needed into a 249-length feature vector, and sends the features into the classifier.

## Output
When the script runs, it:
1. prints the selected data type and dataset path
2. loads all `.bnd` files into memory
3. runs LOSO cross-validation
4. prints:
   - confusion matrix
   - accuracy
   - precision
   - recall
   - F1 score
5. saves the results to a text file inside the `results` directory

Example saved output file:

```text
results/results_o.txt
```

## Current classifier settings
The current `make_classifier()` function builds a Random Forest with:

- `n_estimators=100`
- `criterion="gini"`
- `random_state=4`
- `max_depth=15`
- `min_samples_leaf=5`
