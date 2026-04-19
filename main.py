import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from math import acos, cos, sin
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# python Project1.py z .\FacialLandmarks\BU4DFE_BND_V1.1\
RESULTS_DIR = "./results"
READ_PATH = "./FacialLandmarks/BU4DFE_BND_V1.1/"

@dataclass
class Sample:
  x: np.ndarray # flattened 249-length feature vector
  y: str        # expression label
  group: str    # subject id (folder name)
  path: str     # file path


def parse_args(argv: List[str]) -> Tuple[str, str]:
    """
    Parse and validate command-line arguments.
    Args:
      argv (List[str]): Command-line arguments where argv[1] is the data type
               and argv[2] is the data directory path.
    Returns:
      Tuple[str, str]: A tuple containing (data_type, root_dir) where:
              - data_type (str): The validated data type ('o', 't', 'x', 'y', or 'z')
              - root_dir (str): The validated path to the data directory
    Raises:
      SystemExit: If the number of arguments is not exactly 3 (program name + 2 args).
      ValueError: If data_type is not one of: 'o', 't', 'x', 'y', 'z'.
      FileNotFoundError: If the specified root_dir does not exist or is not a directory.
    """
    if len(argv) != 3:
        print("Usage: python Project1.py <data type to use> <data directory>")
        print("Data types: o (original), t (translated), x, y, z (rotate 180deg)")
        sys.exit(1)

    data_type = argv[1].lower()
    root_dir = argv[2]

    if data_type not in {"o", "t", "x", "y", "z"}:
        raise ValueError("data type must be one of: o, t, x, y, z")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Data directory not found: {root_dir}")

    return data_type, root_dir


def compute_pi() -> float:
    return round(2 * acos(0.0), 3)

PI = compute_pi()

def read_bnd_file(path: str) -> np.ndarray:
  '''
  Read a .bnd file and return the landmark coordinates as a 83 x 3 numpy array.

  Handles cases where the file may contain an extra empty row
  '''
  coords = []
  # open the file
  with open(path, "r") as f:
    # extract the x,y,z coords from each row, ignore index
    for line in f:
      line = line.strip()
      if not line:
        continue

      parts = line.split()
      
      try:
        # the first part is the index, ignore it
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])

      except ValueError:
        continue

      coords.append([x,y,z])

  arr = np.array(coords)
  return arr # shape (83,3)



# Flatten Array
def flatten_features(features) -> np.ndarray:
  """
  Flatten a multi-dimensional features array into a 1D vector.

  Args:
    features (np.ndarray): A multi-dimensional numpy array of features.

  Returns:
    np.ndarray: A flattened 1D vector of the input features with length 249.
  """
  return features.flatten() # returns a vector of length 249

# Translate to origin
def translate_to_origin(features) -> np.ndarray:
  """
  Translates 3D landmarks to the origin by centering them.
  
  Calculates the centroid of all landmarks and subtracts it from each landmark,
  effectively translating the point cloud so that its center is at the origin (0, 0, 0).
  
  Args:
    features (np.ndarray): Array of 3D landmarks with shape (n_landmarks, 3),
                where each row represents (x, y, z) coordinates.
  
  Returns:
    np.ndarray: Centered landmarks with shape (n_landmarks, 3), with the same
           structure as input but translated to have mean at origin.
  """
  # take the average of each landmark (x,y,z) to get the center
  center = np.mean(features, axis=0)
  # subtract the center from each landmark
  return features - center # shape (83,3)


# Rotate around axis by 180 degrees
def rotate_180(features: np.ndarray, axis: str) -> np.ndarray: 
  """
  Rotate 3D landmarks 180 degrees around the specified axis.
  Args:
    features (np.ndarray): Input array of shape (83, 3) containing 3D landmark coordinates.
    axis (str): The axis around which to rotate. Must be one of 'x', 'y', or 'z'.
  Returns:
    np.ndarray: Rotated landmarks with the same shape as input (83, 3).
  Raises:
    ValueError: If axis is not one of 'x', 'y', or 'z'.
  """
  # Given an input of shape (83,3), rotate the landmarks 180 degrees around the specified axis (x,y,z)
  c = cos(PI)
  s= sin(PI)

  if axis == "x":
    R = np.array([[1,0,0],
                  [0,c,s],
                  [0,-s,c]])
  elif axis == "y":
    R = np.array([[c,0,-s],
                  [0,1,0],
                  [s,0,c]])
  elif axis == "z":
    R = np.array([[c,s,0],
                  [-s,c,0],
                  [0,0, 1]])
  else:
      raise ValueError("Axis must be 'x', 'y', or 'z'")

  # Perform matrix multiplication
  rotated_features = np.dot(features, R)
  return rotated_features

def make_classifier():
  """
  Create and return a configured Random Forest classifier.

  Returns:
    RandomForestClassifier: A Random Forest classifier with the following configuration:
      - n_estimators (int): 100 trees in the forest
      - max_depth (int): Maximum depth of trees set to 15
      - min_samples_leaf (int): Minimum samples required at leaf node set to 5
      - n_jobs (int): -1 to use all processors
      - criterion (str): "gini" for split quality measurement
      - random_state (int): 4 for reproducible results
  """
  # build a random forest
  return RandomForestClassifier(
      n_estimators=100,
      max_depth = 15,
      min_samples_leaf=5,
      n_jobs=-1,
      criterion="gini",
      random_state=4, 
  )


def run_loso(samples: List[Sample]):
  """
  Execute Leave-One-Group-Out (LOSO) cross-validation on the provided samples.
  This function performs LOSO validation by splitting the data into training and test sets
  based on sample groups, training a classifier on each fold, and collecting predictions.
  Args:
    samples (List[Sample]): A list of Sample objects, each containing features (x),
                target value (y), and group identifier.
  Returns:
    tuple: A tuple of (y_true, y_pred) where:
         - y_true (list): True target values collected from all test folds
         - y_pred (list): Predicted target values from the classifier on all test folds
  Notes:
    - Uses LeaveOneGroupOut splitter from scikit-learn for data partitioning
    - Trains a new classifier for each fold
    - Timing information is calculated but not currently returned
    - Commented print statements provide fold progress and time estimates
  """
  t_start = time.perf_counter()
  # run loso
  groups = np.array([s.group for s in samples])
  total_folds = len(np.unique(groups))
  X = np.array([s.x for s in samples]) # shape (num_samples, 249)
  y = np.array([s.y for s in samples]) # shape (num_samples,)
  logo = LeaveOneGroupOut() # creates the loso splitter

  y_true = []
  y_pred = []

  t_all0 = time.perf_counter()
  fold_times = []
  for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):

    t0 = time.perf_counter() 

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = make_classifier()
    clf.fit(X_train, y_train)
    y_pred_fold = clf.predict(X_test)

    y_true.extend(y_test)
    y_pred.extend(y_pred_fold)

    dt = time.perf_counter() - t0
    fold_times.append(dt)

    if fold_i == 1 or fold_i % 10 == 0:
        avg = sum(fold_times) / len(fold_times)
        est_total = avg * total_folds
        elapsed = time.perf_counter() - t_all0
        remaining = max(0.0, est_total - elapsed)

        #print(f"[Fold {fold_i}/{total_folds}] fold={dt:.2f}s, avg={avg:.2f}s")
        #print(f"Elapsed: {elapsed/60:.2f} min | ETA remaining: {remaining/60:.2f} min | "
        #      f"Est total: {est_total/60:.2f} min")

  return y_true, y_pred

  # collect all results, precision, accuracy etc.
def print_results(y_true, y_pred):
  """
  Print and return classification model evaluation metrics.
  Calculates and displays accuracy, precision, recall, F1 score, and confusion matrix
  for a classification model's predictions.
  Args:
    y_true: Ground truth (correct) target values.
    y_pred: Estimated target values (model predictions).
  Returns:
    tuple: A tuple containing:
      - accuracy (float): Overall accuracy score.
      - precision (float): Macro-averaged precision score.
      - recall (float): Macro-averaged recall score.
      - f1 (float): Macro-averaged F1 score. (accidently included in return but not printed)
      - confusion_matrix_result (ndarray): Confusion matrix showing predictions vs ground truth.
  """
  # print results in a nice format
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='macro')
  recall = recall_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  confusion_matrix_result = confusion_matrix(y_true, y_pred)

  print("Confusion Matrix:")
  print(confusion_matrix_result)

  print(f"Accuracy: {accuracy:.3f}")
  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  #print(f"F1 Score: {f1:.3f}")

  return accuracy, precision, recall, f1, confusion_matrix_result

def save_results(accuracy, precision, recall, f1, confusion_matrix_result,results_dir=RESULTS_DIR, run_name=None):
  """
  Save model evaluation results to a text file.
  Args:
    accuracy (float): The accuracy score of the model.
    precision (float): The precision score of the model.
    recall (float): The recall score of the model.
    f1 (float): The F1 score of the model (currently unused).
    confusion_matrix_result (list or array): The confusion matrix as a 2D structure.
    results_dir (str, optional): Directory path where results will be saved. Defaults to RESULTS_DIR.
    run_name (str, optional): Name identifier for the results file. Defaults to None.
  Returns:
    None
  Raises:
    OSError: If the results directory cannot be created.
    IOError: If the results file cannot be written.
  Note:
    The F1 score parameter is accepted but not currently written to the output file.
    The function creates the results directory if it does not exist.
    Results are saved to a file named "results_{run_name}.txt".
  """
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  
  run = run_name
  with open(os.path.join(results_dir, f"results_{run}.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"Recall: {recall:.3f}\n")
    #f.write(f"F1 Score: {f1:.3f}\n")
    f.write("Confusion Matrix:\n")
    for row in confusion_matrix_result:
      f.write(f"{row}\n")


def main():
  # Read in files
  data_type, root_dir =parse_args(sys.argv) # get data type and root dir from command line
  print(f"Using data type: {data_type}, from directory: {root_dir}")
  samples = []
  # root is the subjects expression folder: ./FacialLandmarks/BU4DFE_BND_V1.1/M024\Disgust
  for root, dirs, files in os.walk(root_dir):
    # get the expression label from the folder name (last part of root)
    expression_label = os.path.basename(root)
    group = os.path.basename(os.path.dirname(root)) # get the subject id from the parent folder name
    for file in files: 
      if file.endswith(".bnd"):
        path = os.path.join(root, file)
        features_3d = read_bnd_file(path) # shape (83,3)
        if data_type == "o":
          features = flatten_features(features_3d)
        elif data_type == "t":
          features = translate_to_origin(features_3d)
          features = flatten_features(features) # transform to origin then flatten
        elif data_type in {"x", "y", "z"}:
          features = rotate_180(features_3d, axis=data_type)
          features = flatten_features(features) # rotate 180 degrees around the specified axis then flatten
        else:
          raise ValueError("Invalid data type")
        samples.append(Sample(x=features, y=expression_label, group=group, path=path))

  # groups contains the subject ids for each sample, labels contains the expression labels for each sample
  groups = np.array([s.group for s in samples])
  labels = np.array([s.y for s in samples])

  print("Unique groups (subjects):", len(np.unique(groups)))
  print("Unique labels (expressions):", len(np.unique(labels)))
  print("Example (group, label) pairs:", list(zip(groups[:10], labels[:10])))

  # Run LOSO
  print(f"Running LOSO cross-validation on {len(samples)} samples...\n")
  y_true, y_pred = run_loso(samples)

  # Print results
  accuracy, precision, recall, f1, confusion_matrix_result = print_results(y_true, y_pred)

  # Save results
  save_results(accuracy, precision, recall, f1, confusion_matrix_result, results_dir=RESULTS_DIR, run_name=data_type)

if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"Total execution time: {time_end - time_start:.2f} seconds")
