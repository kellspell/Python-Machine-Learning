# Python Machine Learning — Study Book

A personal study guide covering the mathematical and statistical foundations of Machine Learning.

---

## Table of Contents

- [Chapter 1 — Probability Basics](Probability/Probability.md)
  - General Definition of Probability
  - Bayes Theorem
  - Common Probability Distributions (Gaussian, Bernoulli, Binomial, Poisson)
  - Measures of Central Tendency and Dispersion (Mean, Median, Mode)
  - Hypothesis Testing
  - Confidence Intervals and Statistical Significance

- [Chapter 2 — Probability Deep Dive](Probability-02/ReadMe.md)
  - Sample Space and Events
  - Random Variables (PMF & PDF)
  - Expectation, Variance, and Standard Deviation
  - Common Distributions and their ML Applications
  - Visualizing Distributions (Skewness & Kurtosis)
  - Statistical Inference
  - Point and Interval Estimation
  - Constructing Confidence Intervals

- [Chapter 3 — Hypothesis Testing & Statistics](Hypotsis/ReadMe.md)
  - Introduction to Hypothesis Testing
  - P-values and Significance Levels
  - Types of Errors (Type I & Type II)
  - T-Tests (One-Sample, Two-Sample, Paired)
  - Chi-Square Test
  - ANOVA — Analysis of Variance
  - Understanding Correlation (Pearson & Spearman)
  - Linear Regression Basics
  - Interpreting Regression Results (Slope, Intercept, R-Squared)

- [Chapter 4 — Machine Learning Foundations](ML/Python-MML.md)
  - Types of Machine Learning (Supervised, Unsupervised, Reinforcement)
  - Key Concepts: Features, Target, Overfitting, Underfitting, Bias-Variance Tradeoff
  - Integrals and their Applications in ML
  - Optimization Concepts (Local vs Global Minima, Convex vs Non-Convex)
  - Stochastic Gradient Descent (SGD) and Variants (Mini-batch, Momentum, Adam)
  - Overview of Supervised Learning
  - Introduction to Regression Analysis
  - Cost Function and Optimization in Linear Regression
  - Polynomial Regression for Non-Linear Relationships
  - Regularization Techniques — Lasso (L1) and Ridge (L2)
  - Classification Problems and Common Use Cases
  - Logistic Regression for Binary Classification
  - Sigmoid Function, Decision Boundary, and Coefficient Interpretation
  - Model Evaluation Metrics for Regression (MSE, MAE, RMSE)
  - Model Evaluation Metrics for Classification (Accuracy, Precision, Recall, F1-Score)
  - Introduction to Cross-Validation (K-Fold, Stratified K-Fold, LOOCV)

- [Chapter 5 — Feature Engineering & Model Evaluation](Feature-Engineering-Model-Eval/README.md)
  - What is Feature Engineering and Why it Matters
  - Raw Data vs. Engineered Features
  - Feature Engineering Techniques (Combining, Binning, Interaction, Aggregation, Polynomial)
  - Scaling and Normalization (Min-Max Scaling, Standardization)
  - Encoding Categorical Variables (One-Hot Encoding, Label Encoding)
  - Dealing with High-Cardinality Categorical Features (Frequency & Target Encoding)
  - When to Use Each Encoding Technique
  - Feature Creation (Date-Time Features, Interaction Features, Aggregation Features)
  - Feature Transformation (Log, Square Root, Polynomial)
  - Evaluation Metrics for Regression (MAE, MSE, RMSE, R-Squared)
  - Evaluation Metrics for Classification (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
  - When to Use Each Metric
  - Cross-Validation (K-Fold, Stratified K-Fold, LOOCV)
  - Hyperparameter Tuning (Grid Search, Random Search)

- [Chapter 6 — Machine Learning Algorithms & Ensemble Learning](Machine-Learn-Algo/ReadMe.md)
  - Introduction to Ensemble Learning (Bagging, Boosting, Stacking)
  - Why Ensembles Outperform Single Models (Bias-Variance-Robustness)
  - Bagging (Bootstrap Aggregating) and the Detective Team Analogy
  - Random Forest — Key Parameters, Advantages, and Feature Importance
  - Boosting — The Sequential Learning Approach
  - Gradient Boosting (Step-by-Step, Key Parameters, Regularization)
  - XGBoost — Speed, Efficiency, and Advanced Hyperparameter Tuning
  - LightGBM — Leaf-Wise Growth and Handling Large Datasets
  - CatBoost — Native Categorical Feature Handling
  - XGBoost vs LightGBM vs CatBoost — Detailed Comparison and Decision Guide
  - Imbalanced Data — Techniques (SMOTE, Undersampling, Class Weights) and Evaluation Metrics

- [Chapter 7 — Model Tuning & Bayesian Optimization](Model-Tuning-Optimization/README.md)
  - Introduction to Bayesian Optimization and Why It Outperforms Grid/Random Search
  - The Core Idea: Surrogate Model and Acquisition Function
  - How Bayesian Optimization Works Step-by-Step (Random Init → Surrogate → Acquisition → Repeat)
  - Prediction vs Uncertainty in the Surrogate Model
  - Exploration vs Exploitation Trade-off
  - Acquisition Function (Expected Improvement and variants)
  - Hyperopt Library — Tree-Structured Parzen Estimator (TPE)
  - Optuna Library — Define-by-Run API and Pruning

- [Chapter 8 — Neural Networks & Deep Learning](Neural%20Networks,%20Deep%20Learning/README.md)
  - What is Deep Learning and How it Differs from Classical ML
  - Artificial Neural Networks (ANNs) — Structure, Neurons, Layers, Weights, Biases
  - Forward Propagation — How Data Flows Through the Network
  - Activation Functions — Sigmoid, Tanh, ReLU, Softmax, Linear
  - Loss Functions — MSE, Cross-Entropy, Binary Cross-Entropy
  - Backpropagation — Gradients, Chain Rule, and Weight Updates
  - TensorFlow & Keras — Sequential and Functional API, Model Compilation, Training
  - PyTorch — Tensors, Autograd, torch.nn, The 5-Step Training Loop
  - Saving, Loading, and Evaluating Models in Both Frameworks
  - PyTorch vs TensorFlow/Keras — Comparison and When to Use Each

- [Chapter 9 — Convolutional Neural Networks (CNNs)](CNN/READEME.md)
  - What are CNNs and Why They Excel at Image Processing
  - CNN Architecture — Convolutional Layer, ReLU, Pooling, Fully Connected Layer
  - Kernel Size — Small vs Large Kernels and Feature Detection Trade-offs
  - Strides — How Step Size Affects Output Dimensions
  - Padding — Valid vs Same Padding and the Border Problem
  - How Convolution Extracts Features — Edge, Texture, and Abstract Pattern Hierarchy
  - Pooling Layers — Max Pooling vs Average Pooling and Dimensionality Reduction
  - Translation Invariance, Parameter Efficiency, and Automatic Feature Extraction
  - Real-World CNN Applications (Image Classification, Object Detection, etc.)

- [Chapter 10 — RNNs & Sequence Modeling](RNNs-Sequence-Modeling/README.md)
  - What is Sequence Modeling and Why Order Matters
  - Recurrent Neural Networks (RNNs) — Structure, Hidden State, and Unrolled View
  - Backpropagation Through Time (BPTT) — Forward Pass, Loss, and Gradient Flow
  - Vanishing & Exploding Gradient Problems and Solutions (Gradient Clipping)
  - LSTM — Forget Gate, Input Gate, Output Gate, and Cell State
  - GRU — Update Gate, Reset Gate, and Candidate Hidden State
  - GRU vs LSTM — Architecture, Speed, and When to Use Each
  - Text Preprocessing — Tokenization, Stopwords, Stemming, Lemmatization
  - Word Embeddings — Dense Vectors, Word2Vec, GloVe, Pre-trained Embeddings
  - Sequence-to-Sequence Models — Encoder-Decoder Architecture
  - Attention Mechanism — Context Vector, Score Functions, and Spotlight Analogy
