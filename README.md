  
# Task 01: Classification Fundamentals and MNIST Digit Recognition

# Task 02: Regression Fundamentals and California Housing Price Prediction

## 📈 Task 02 Overview

This section covers **Task 02: Regression Fundamentals and California Housing Price Prediction** for the ML Internship at ARCH Technologies. The project applies Chapter 2 regression concepts to predict housing prices using the California Housing dataset.

## 📋 Task 02 Requirements

✅ **All requirements successfully completed:**
- [x] Chapter 2 study and exercises
- [x] California Housing dataset loading and preprocessing
- [x] Linear Regression and Decision Tree Regression model training
- [x] Achieve RMSE ≤ 0.7 on test set
- [x] Performance comparison tables (Linear vs Tree)
- [x] Error analysis and feature importance
- [x] Model improvement (feature engineering or ensemble)
- [x] Gradio web application deployment

## 🏆 Task 02 Results Summary

| Metric         | Linear Regression | Decision Tree | Target   |
|----------------|------------------|--------------|----------|
| **Test RMSE**  | 0.74             | **0.65** ✅   | ≤0.7     |
| **Training Time** | ~2s           | ~10s         | -        |
| **Status**     | Slightly above   | **Target Achieved** | ✅ |

## 🚀 Task 02 Quick Start

1. Install dependencies (already included in requirements.txt)
2. Run `Chapter2-Regression.ipynb` for full workflow
3. Launch Gradio app: `python app_regression.py`

## 📁 Task 02 Repository Structure

```
├── Chapter2-Regression.ipynb         # Regression implementation notebook
├── Task02_Regression_Report.md       # Project report for Task 02
├── app_regression.py                 # Gradio web application for regression
```

## Key Results:
- End-to-end California Housing price prediction pipeline
- Multiple regression model comparisons and evaluations
- Error analysis and feature importance insights
- Deployable web application for price prediction
- All Task 02 requirements satisfied


## 🎯 Project Overview

This repository contains the complete implementation of **Task 01: Classification Fundamentals and MNIST Digit Recognition** for the ML Internship at ARCH Technologies. The project demonstrates mastery of Chapter 3 classification concepts through practical MNIST digit recognition implementation.

## 📋 Task Requirements

✅ **All requirements successfully completed:**
- [x] Chapter 3 study and exercises
- [x] MNIST dataset implementation (60k train, 10k test)
- [x] SGD and Random Forest classifier training
- [x] Achieve ≥95% test accuracy
- [x] Performance comparison tables (SGD vs RF, OvR vs OvO)
- [x] Error analysis with pattern identification
- [x] Model improvement implementation
- [x] Gradio web application deployment

## 🏆 Results Summary

| Metric | SGD Classifier | Random Forest | Target |
|--------|----------------|---------------|---------|
| **Test Accuracy** | 91.2% | **97.1%** ✅ | ≥95% |
| **Training Time** | ~5s | ~45s | - |
| **Status** | Below target | **Target Achieved** | ✅ |

## 📁 Repository Structure

```
├── Chapter3-Classification.ipynb     # Main implementation notebook
├── Task01_Classification_Report.md   # Comprehensive project report  
├── app.py                           # Gradio web application
├── task01_demo.py                   # Standalone demonstration script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook Chapter3-Classification.ipynb
```

### 3. Launch Web Application
```bash
python app.py
```
- Error Analysis with systematic error identification

#### 2. MNIST Digit Recognition Project ✅
- **(a)** Load MNIST dataset ✅
- **(b)** Split data (60k train, 10k test) ✅
- **(c)** Train classifiers:
  - SGD Classifier (with hinge loss) ✅
  - Random Forest Classifier ✅
- **(d)** Evaluate using confusion matrix & classification report ✅
- **(e)** Visualize errors (plot worst misclassifications) ✅
- **(f)** Deploy as Gradio web app ✅
- **Target**: Achieve minimum 95% test accuracy ✅

#### 3. Error Analysis Report ✅
- Identify 3 common error patterns ✅
- Propose solutions (data augmentation, preprocessing) ✅
- Implement one improvement (ensemble method) and measure impact ✅

#### 4. Comparison Tables ✅
- SGD Classifier vs Random Forest performance ✅
- OvR vs OvO strategies for multiclass ✅

### Expected Deliverables:

#### PDF Report ✅
- Chapter 3 exercise solutions + MNIST Digit Recognition Project
- Error analysis findings
- Training/validation curves
- Performance comparison tables

#### GitHub Repository ✅
- Jupyter notebooks (data exploration + training)
- Gradio app code (app.py + requirements.txt)
- Standalone demo script

### How to Run:

1. **Notebook**: Open `Chapter3-Classification.ipynb` and run cells sequentially
2. **Demo Script**: `python task01_demo.py`
3. **Web App**: `python app.py` (after running notebook to generate model files)

### Key Results:
- Comprehensive MNIST digit classification implementation
- Multiple classifier comparisons and evaluations
- Error analysis with improvement strategies
- Deployable web application for digit recognition
- All Task 01 requirements satisfied
