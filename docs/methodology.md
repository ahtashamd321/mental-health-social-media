Methodology Documentation
Mental Health & Social Media Balance Analytics
Author: Ahtasham Anjum
Date: November 2024
Version: 1.0

Table of Contents

Data Collection Process
Statistical Methods
Machine Learning Model Selection
Validation Approach
Feature Engineering
Ethical Considerations


1. Data Collection Process
1.1 Dataset Overview

Source: Mental Health and Social Media Balance Dataset
Collection Period: 2024
Sample Size: 500 users
Sampling Method: Stratified random sampling across age groups and platforms
Data Format: CSV (Comma-Separated Values)

1.2 Variables Collected
Independent Variables:

Age: Participant age (16-49 years)
Gender: Male, Female, Other
Daily Screen Time: Self-reported hours per day (1.0-11.0 hours)
Sleep Quality: Self-assessed on 1-10 scale
Stress Level: Self-reported on 1-10 scale
Days Without Social Media: Monthly frequency of digital breaks (0-9 days)
Exercise Frequency: Weekly exercise days (0-7 days)
Social Media Platform: Primary platform used (Facebook, Instagram, TikTok, LinkedIn, YouTube, Twitter)

Dependent Variable:

Happiness Index: Self-reported overall happiness (1-10 scale)

1.3 Data Quality Assurance

Missing Values: None detected (100% completion rate)
Duplicates: Zero duplicate entries found
Outlier Detection: Applied IQR method; no significant outliers removed
Data Validation:

Range checks for all numerical variables
Consistency checks for categorical variables
Cross-validation of self-reported metrics



1.4 Ethical Considerations

All data anonymized (User IDs: U001-U500)
No personally identifiable information (PII) collected
Self-reported metrics to maintain privacy
Dataset used for educational and research purposes only


2. Statistical Methods
2.1 Descriptive Statistics
Measures Applied:

Central Tendency: Mean, median, mode
Dispersion: Standard deviation, variance, range, IQR
Distribution Analysis: Skewness, kurtosis

Purpose: Understand the basic characteristics and distribution of all variables.
2.2 Correlation Analysis
Pearson Correlation Coefficient

Method: Parametric correlation for linear relationships
Formula: r = Cov(X,Y) / (σ_X × σ_Y)
Interpretation:

r > 0.7: Strong positive correlation
0.4 < r < 0.7: Moderate positive correlation
r < 0.4: Weak correlation


Significance Level: α = 0.05
Null Hypothesis: No correlation exists (r = 0)

Spearman Rank Correlation

Method: Non-parametric correlation for monotonic relationships
Use Case: Backup validation for non-normal distributions
Applied to: All continuous variables

2.3 Hypothesis Testing
Analysis of Variance (ANOVA)

Type: One-way ANOVA
Purpose: Compare means across multiple groups
Applied To:

Platform comparison (6 platforms)
Age group comparison (4 groups)
Screen time categories (5 groups)



Hypotheses:

H₀: μ₁ = μ₂ = ... = μₖ (no difference in means)
H₁: At least one mean differs significantly

Test Statistic: F = Between-group variance / Within-group variance
Significance Level: α = 0.05
Post-hoc Test: Tukey HSD for pairwise comparisons (when ANOVA is significant)
Independent T-Tests

Purpose: Compare means between two groups
Applied To:

Gender differences (Male vs Female)
Binary comparisons



Assumptions Checked:

Independence of observations
Normal distribution (Shapiro-Wilk test)
Homogeneity of variances (Levene's test)

2.4 Effect Size Calculations

Cohen's d: For t-tests
Eta-squared (η²): For ANOVA
Interpretation:

Small: d = 0.2, η² = 0.01
Medium: d = 0.5, η² = 0.06
Large: d = 0.8, η² = 0.14




3. Machine Learning Model Selection
3.1 Problem Formulation
Task Type: Supervised Learning - Regression
Target Variable: Happiness Index (continuous, 1-10 scale)
Objective: Predict happiness based on social media usage patterns and lifestyle factors
3.2 Models Evaluated
Model 1: Linear Regression
Algorithm: Ordinary Least Squares (OLS)
Assumptions Tested:

Linearity: Residual plots
Independence: Durbin-Watson test
Homoscedasticity: Breusch-Pagan test
Normality of residuals: Q-Q plot, Shapiro-Wilk test

Hyperparameters: None (baseline model)
Results:

R² Score: 0.72
MAE: 0.85
RMSE: 1.08
Training Time: < 1 second

Advantages:

Interpretable coefficients
Fast training
Low computational cost

Limitations:

Assumes linear relationships
Sensitive to outliers
Cannot capture complex interactions


Model 2: Random Forest Regressor ⭐ (Selected)
Algorithm: Ensemble of Decision Trees with Bootstrap Aggregating
Hyperparameter Tuning:
pythonn_estimators: [50, 100, 150, 200]
max_depth: [5, 10, 15, 20, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
max_features: ['auto', 'sqrt', 'log2']
Optimal Hyperparameters (Grid Search):

n_estimators: 100
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
max_features: 'auto'
random_state: 42

Results:

R² Score: 0.76 ⭐
MAE: 0.73
RMSE: 0.98
Cross-Validation R²: 0.74 ± 0.03
Training Time: ~3 seconds

Advantages:

Handles non-linear relationships
Robust to outliers
Feature importance metrics
No scaling required
Minimal overfitting

Limitations:

Less interpretable than linear models
Longer training time
Larger model size

Why Selected:

Best cross-validation performance
Stable predictions across folds
Provides interpretable feature importance
Generalizes well to unseen data


Model 3: Gradient Boosting Regressor
Algorithm: Sequential ensemble with gradient descent optimization
Hyperparameters:

n_estimators: 100
learning_rate: 0.1
max_depth: 5
subsample: 0.8
random_state: 42

Results:

R² Score: 0.75
MAE: 0.76
RMSE: 1.00
Training Time: ~5 seconds

Advantages:

High accuracy
Handles missing data
Feature importance

Limitations:

Longer training time
Risk of overfitting
Requires careful tuning

Why Not Selected:

Slightly lower R² than Random Forest
Higher computational cost
More prone to overfitting


3.3 Model Selection Criteria
CriterionWeightLinear RegRandom Forest ⭐Gradient BoostR² Score30%0.720.760.75Cross-Val Stability25%HighHighestMediumInterpretability20%HighMediumLowTraining Speed15%FastestFastSlowGeneralization10%GoodBestGoodOverall Score100%72%87%79%
Final Selection: Random Forest Regressor

4. Validation Approach
4.1 Train-Test Split
Method: Stratified random split

Training Set: 80% (400 samples)
Test Set: 20% (100 samples)
Random State: 42 (for reproducibility)
Stratification: By happiness index quartiles

Rationale: 80-20 split provides sufficient training data while maintaining adequate test set for validation.
4.2 Cross-Validation
Method: K-Fold Cross-Validation

K: 5 folds
Shuffle: True
Random State: 42

Process:

Dataset divided into 5 equal parts
Model trained on 4 folds, validated on 1
Process repeated 5 times (each fold used as validation once)
Average performance calculated

Metrics Recorded:

R² score per fold
Mean R²: 0.74
Standard Deviation: ±0.03
Min R²: 0.71
Max R²: 0.77

Interpretation: Low standard deviation indicates stable, consistent model performance.
4.3 Performance Metrics
Regression Metrics:
1. R² Score (Coefficient of Determination)

Formula: R² = 1 - (SS_res / SS_tot)
Range: 0 to 1 (higher is better)
Interpretation: Proportion of variance explained
Result: 0.76 (76% variance explained)

2. Mean Absolute Error (MAE)

Formula: MAE = (1/n) Σ|y_i - ŷ_i|
Unit: Same as target variable (1-10 scale)
Interpretation: Average prediction error
Result: 0.73 points

3. Root Mean Squared Error (RMSE)

Formula: RMSE = √[(1/n) Σ(y_i - ŷ_i)²]
Unit: Same as target variable
Interpretation: Penalizes large errors more
Result: 0.98 points

4. Mean Squared Error (MSE)

Formula: MSE = (1/n) Σ(y_i - ŷ_i)²
Result: 0.96

4.4 Residual Analysis
Tests Performed:

Residual Plot: Check for homoscedasticity

Result: No clear pattern detected ✓


Q-Q Plot: Check for normality

Result: Residuals approximately normal ✓


Histogram of Residuals: Distribution check

Result: Centered around zero ✓


Predicted vs Actual Plot: Model fit

Result: Points closely follow diagonal ✓



4.5 Feature Importance Validation
Method: Permutation Importance

Shuffle each feature and measure performance drop
Validates Random Forest's built-in feature importance
Confirms top predictors: Sleep Quality, Stress Level, Screen Time


5. Feature Engineering
5.1 Created Features
1. Age Groups
Method: Binning continuous age variable
16-20: Young Adults
21-30: Millennials
31-40: Mid-Career
41-50: Mature Professionals
Purpose: Capture generational differences
2. Screen Time Categories
Method: Binning daily screen time
1-3 hrs: Low Usage
3-5 hrs: Moderate Usage
5-7 hrs: High Usage
7-9 hrs: Very High Usage
9+ hrs: Excessive Usage
Purpose: Identify usage patterns and optimal ranges
3. Wellbeing Score
Formula:
Wellbeing = (Sleep_Quality × 0.3) + 
            ((10 - Stress_Level) × 0.3) + 
            (Happiness_Index × 0.2) + 
            ((10 - Screen_Time) × 0.1) + 
            (Exercise_Frequency × 0.1)
Purpose: Composite metric for user segmentation
4. User Segments
Method: Wellbeing score quartiles
Critical: Score < 5.0
At Risk: 5.0 ≤ Score < 6.5
Balanced: 6.5 ≤ Score < 8.0
Thriving: Score ≥ 8.0
Purpose: Identify intervention priorities
5.2 Encoding Categorical Variables
Method: Label Encoding

Gender: Male=0, Female=1, Other=2
Platform: Alphabetical order (0-5)

Rationale: Tree-based models handle label encoding effectively
5.3 Feature Scaling
Method: StandardScaler (for Linear Regression only)

Formula: z = (x - μ) / σ
Applied: Linear Regression model
Not Applied: Tree-based models (not required)


6. Ethical Considerations
6.1 Data Privacy

Anonymization: All user identifiers removed
No PII: No names, emails, or contact information
Secure Storage: Dataset stored securely with access controls

6.2 Bias Mitigation
Strategies:

Balanced representation across gender and age groups
Multiple platform inclusion to avoid single-platform bias
Self-reported metrics to reduce observer bias

Limitations Acknowledged:

Self-reported data may contain response bias
Cross-sectional design cannot establish causation
Sample may not represent all demographics

6.3 Responsible AI
Principles Followed:

Transparency: Full methodology disclosed
Reproducibility: Random seeds set, code documented
Fairness: Equal treatment across demographic groups
Accountability: Results presented with limitations

6.4 Intended Use
Appropriate Use:

Educational portfolio project
Research insights into digital wellbeing
Awareness about social media impact
Personal wellness optimization

Inappropriate Use:

Clinical diagnosis or treatment
Individual psychological assessment
Discriminatory decision-making
Commercial profiling without consent


7. Limitations & Future Work
7.1 Current Limitations

Cross-sectional Design: Cannot establish causality
Self-reported Data: Potential recall and social desirability bias
Sample Size: 500 users; larger sample could improve generalization
Platform Diversity: Limited to 6 major platforms
Temporal Factors: No time-series data to capture trends

7.2 Future Improvements
Short-term (1-3 months):

 Collect longitudinal data (track users over time)
 Expand sample size to 1000+ users
 Add qualitative interviews for context

Medium-term (3-6 months):

 Implement causal inference methods (propensity score matching)
 Develop deep learning models
 Add NLP analysis of user comments

Long-term (6-12 months):

 Real-time data collection pipeline
 Integration with wearable device data
 Multi-language support for global reach


8. References
Statistical Methods

Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics. Sage Publications.
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Routledge.

Machine Learning

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Digital Wellbeing Research

Twenge, J. M., & Campbell, W. K. (2018). Associations between screen time and lower psychological well-being among children and adolescents. Preventive Medicine, 12, 271-283.
Hunt, M. G., et al. (2018). No More FOMO: Limiting Social Media Decreases Loneliness and Depression. Journal of Social and Clinical Psychology, 37(10), 751-768.


Appendix
A. Software & Libraries
Programming Language: Python 3.9+
Core Libraries:

pandas 2.1.3: Data manipulation
numpy 1.24.3: Numerical computing
scikit-learn 1.3.2: Machine learning
scipy 1.11.4: Statistical tests
matplotlib 3.8.2: Visualization
seaborn 0.13.0: Statistical plots
plotly 5.18.0: Interactive charts
streamlit 1.29.0: Web dashboard

Development Tools:

Jupyter Notebook: Interactive analysis
Git: Version control
GitHub: Repository hosting

B. Reproducibility
Random Seeds Set:

Train-test split: random_state=42
Cross-validation: random_state=42
Random Forest: random_state=42
All stochastic processes: seed=42

Hardware Used:

CPU: Intel Core i7
RAM: 16GB
OS: Windows 11 / macOS / Linux

Execution Time:

Data loading: < 1 second
EDA: ~30 seconds
Model training: ~5 seconds
Total pipeline: < 1 minute


Document Version: 1.0
Last Updated: November 2024
Author: Ahtasham Anjum
Contact: ahtashamd321@gmail.com
GitHub: github.com/ahtashamd321/mental-health-social-media