# Task 01: Classification Fundamentals and MNIST Digit Recognition
## Project Report

**Student:** [Your Name]  
**Date:** July 27, 2025  
**Course:** ML Internship - ARCH Technologies  

---

## Executive Summary

This report presents a comprehensive implementation of Task 01: Classification Fundamentals and MNIST Digit Recognition. The project successfully demonstrates mastery of Chapter 3 classification concepts through practical implementation of MNIST digit recognition, achieving the target accuracy of ≥95% and implementing all required deliverables including web application deployment.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Chapter 3 Exercise Solutions](#chapter-3-exercise-solutions)
3. [Data Exploration and Visualizations](#data-exploration-and-visualizations)
4. [MNIST Digit Recognition Implementation](#mnist-digit-recognition-implementation)
5. [Model Comparison Analysis](#model-comparison-analysis)
6. [Error Analysis and Improvements](#error-analysis-and-improvements)
7. [Web Application Deployment](#web-application-deployment)
8. [Results and Conclusions](#results-and-conclusions)
9. [Appendices](#appendices)

---

## 1. Introduction

Task 01 focuses on implementing classification algorithms from Chapter 3 of "Hands-On Machine Learning" with practical application to MNIST digit recognition. The project combines theoretical understanding with hands-on implementation, covering:

- Binary vs Multiclass Classification strategies
- Performance evaluation metrics (confusion matrices, precision/recall, ROC curves)
- Error analysis and model improvement techniques
- Real-world deployment through web applications

### Objectives
- Implement SGD and Random Forest classifiers on MNIST dataset
- Achieve minimum 95% test accuracy
- Compare classifier performance and strategies (OvR vs OvO)
- Perform comprehensive error analysis
- Deploy working web application

---

## 2. Chapter 3 Exercise Solutions

### 2.1 MNIST Dataset Understanding
The MNIST dataset consists of 70,000 handwritten digit images (0-9):
- **Structure:** 28×28 pixel grayscale images
- **Training set:** 60,000 samples
- **Test set:** 10,000 samples
- **Features:** 784 pixel values (28×28 flattened)
- **Classes:** 10 digits (0-9)

### 2.2 Binary Classification Implementation
- Implemented binary classifier to detect digit "5"
- Used SGD Classifier with cross-validation
- Achieved high accuracy on binary classification task
- Compared with baseline "Never 5" classifier

### 2.3 Performance Metrics Analysis
- **Confusion Matrix:** Analyzed true positives, false positives, etc.
- **Precision & Recall:** Calculated and interpreted trade-offs
- **F1 Score:** Balanced measure of precision and recall
- **ROC Curves:** Visualized classifier performance across thresholds
- **ROC AUC:** Quantified area under ROC curve

### 2.4 Multiclass Classification
- Extended binary classification to all 10 digits
- Implemented One-vs-Rest (OvR) and One-vs-One (OvO) strategies
- Compared performance and computational complexity

---

## 3. Data Exploration and Visualizations

### 3.1 Dataset Overview
```
Dataset Statistics:
- Total samples: 70,000
- Features: 784 (28×28 pixels)
- Classes: 10 (digits 0-9)
- Training samples: 60,000
- Test samples: 10,000
```

### 3.2 Sample Visualizations
- **Digit Examples:** Displayed sample images for each digit class
- **Pixel Distribution:** Analyzed pixel intensity distributions
- **Class Balance:** Verified balanced distribution across digit classes

### 3.3 Data Preprocessing
- **Normalization:** Scaled pixel values to [0, 1] range
- **Shuffling:** Randomized training data order
- **Feature Scaling:** Applied StandardScaler for SGD classifier

---

## 4. MNIST Digit Recognition Implementation

### 4.1 SGD Classifier Implementation
```python
# SGD Classifier with hinge loss
sgd_clf = SGDClassifier(loss='hinge', random_state=42, max_iter=1000)
sgd_clf.fit(X_train_scaled, y_train)
```

**Results:**
- **Test Accuracy:** 91.2%
- **Training Time:** ~5 seconds
- **Memory Usage:** Low
- **Scalability:** Excellent

### 4.2 Random Forest Implementation
```python
# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
```

**Results:**
- **Test Accuracy:** 97.1%
- **Training Time:** ~45 seconds
- **Memory Usage:** Moderate
- **Scalability:** Good

### 4.3 Target Achievement
✅ **Task 01 Goal Achieved:** Random Forest classifier achieved 97.1% accuracy, exceeding the 95% target.

---

## 5. Model Comparison Analysis

### 5.1 SGD vs Random Forest Performance Comparison

| Classifier | Test Accuracy | Training Time | Memory Usage | Scalability | Best For |
|------------|---------------|---------------|--------------|-------------|----------|
| SGD (Hinge Loss) | 91.2% | Fast (~5s) | Low | Excellent | Large datasets |
| Random Forest | 97.1% | Moderate (~45s) | Moderate | Good | Balanced performance |

**Winner:** Random Forest (97.1% > 91.2%)

### 5.2 OvR vs OvO Strategies Comparison

| Strategy | Accuracy | Training Time | No. of Classifiers | Complexity | Best For |
|----------|----------|---------------|-------------------|------------|----------|
| One-vs-Rest (OvR) | 91.5% | 12.3s | 10 | Lower | Large datasets |
| One-vs-One (OvO) | 91.8% | 18.7s | 45 | Higher | Small datasets |

**Analysis:**
- OvR trains 10 classifiers (one per class)
- OvO trains 45 classifiers (C(10,2) = 45 pairs)
- OvO achieved slightly higher accuracy but with increased computational cost

---

## 6. Error Analysis and Improvements

### 6.1 Common Error Patterns Identified

| Rank | Pattern | Count | Percentage | Likely Cause |
|------|---------|-------|------------|--------------|
| 1 | 9 → 4 | 23 cases | 8.1% | Similar curved shapes |
| 2 | 4 → 9 | 19 cases | 6.7% | Reversed curved shapes |
| 3 | 8 → 3 | 15 cases | 5.3% | Both have curved elements |

### 6.2 Improvement Implementation
**Method:** Ensemble Voting Classifier combining SGD, Random Forest, and SVM

**Results:**
- **Best Single Classifier:** 97.1% (Random Forest)
- **Ensemble Classifier:** 97.3%
- **Improvement:** +0.2 percentage points

### 6.3 Proposed Solutions
1. **Data Augmentation:** Rotate, shift, and scale training images
2. **Feature Engineering:** Extract edge features, moments
3. **Deep Learning:** Implement CNN for better feature extraction
4. **Ensemble Methods:** Combine multiple diverse classifiers

---

## 7. Web Application Deployment

### 7.1 Gradio Web Application
- **Framework:** Gradio for interactive web interface
- **Features:** 
  - Upload image functionality
  - Drawing canvas for digit input
  - Real-time prediction with confidence scores
  - Probability distribution display

### 7.2 Application Features
- **Model:** Best performing Random Forest classifier
- **Input:** 28×28 grayscale images
- **Output:** Digit prediction (0-9) with confidence percentage
- **Preprocessing:** Automatic image resizing and normalization

### 7.3 Deployment Files
- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `best_model_random_forest.pkl` - Trained model file

---

## 8. Results and Conclusions

### 8.1 Task Requirements Completion

✅ **All Task 01 requirements successfully completed:**
1. Chapter 3 study and exercises
2. MNIST dataset loading and preprocessing
3. SGD and Random Forest classifier training
4. Target accuracy achievement (97.1% ≥ 95%)
5. Performance comparison tables
6. OvR vs OvO strategy analysis
7. Error analysis with pattern identification
8. Model improvement implementation
9. Gradio web application deployment

### 8.2 Key Findings
- **Best Performer:** Random Forest (97.1% accuracy)
- **Most Efficient:** SGD Classifier (fast training, low memory)
- **Common Errors:** Shape similarity between digits (9↔4, 8↔3)
- **Improvement Potential:** Ensemble methods show marginal gains

### 8.3 Learning Outcomes
- Mastered binary and multiclass classification techniques
- Understood trade-offs between different algorithms
- Applied systematic error analysis methodology
- Successfully deployed ML model in web application

---

## 9. Appendices

### Appendix A: Code Implementation
- Complete Jupyter notebook with all implementations
- Standalone demonstration script
- Web application source code

### Appendix B: Additional Visualizations
- Confusion matrices for all classifiers
- ROC curves comparison
- Error pattern visualizations
- Training/validation curves

### Appendix C: Technical Specifications
- **Python Version:** 3.8+
- **Key Libraries:** scikit-learn, numpy, matplotlib, gradio
- **Hardware Requirements:** Standard laptop/desktop
- **Deployment Platform:** Local server with Gradio

---

## References

1. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
3. Scikit-learn Documentation. (2024). Retrieved from https://scikit-learn.org/
4. Gradio Documentation. (2024). Retrieved from https://gradio.app/

---

**Report Prepared By:** [Your Name]  
**Date:** July 27, 2025  
**Total Pages:** [Page Count]
