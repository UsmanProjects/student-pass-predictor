# Student Pass/Fail Predictor

A simple machine learning project that predicts whether a student will
Pass or Fail based on hours studied, using the K-Nearest Neighbors (KNN)
algorithm.

## How it works
- Takes "hours studied" as input
- Looks at the 3 closest matching students (K=3)
- Predicts Pass or Fail based on majority vote

## Tech used
- Python
- Scikit-Learn

## How to run
```bash
pip install scikit-learn
python predictor.py
```

## What I learned
- KNN stores data and does its real work at prediction time
- Feature scaling matters because KNN uses distance calculations
- Choosing K affects whether the model overfits or underfits
