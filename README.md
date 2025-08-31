# Longitudinal Prediction Framework for N-Wave Cohort Survey Data

## Project Overview

This repository contains a comprehensive framework for predicting outcomes using machine learning approaches on longitudinal data from multi-wave cohort studies. The implementation is based on the methodology described in:

Guo, X., Liu, S., Jiang, L., Xiong, Z., Wang, L., Lu, L., Li, X., Zhao, L., & Shek, D. T. L. (2025). Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents: A prospective multicenter Cohort study. *Journal of Affective Disorders*, 120110.

## Repository Structure

```
.
├── consolidated_nssi_prediction.py  # Complete framework implementation
├── README.md                       # This file
├── requirements.txt                 # Python dependencies
└── Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents_A prospective multicenter Cohort study.pdf
```

## Framework Architecture

### 1. Data Processing Pipeline

#### Phase 1: Data Transformation
- **Format Conversion**: Transform wide-format data to long-format for longitudinal analysis
- **Variable Organization**: Restructure variables across N time waves (T1, T2, T3, T4, ...)
- **Data Validation**: Check data integrity and consistency

#### Phase 2: Data Cleaning
- **Target Variable Processing**: Convert outcome measures to binary variables (0/1)
- **Quality Control**: Remove invalid observations and apply inclusion criteria
- **Missing Value Assessment**: Comprehensive analysis of missing data patterns

#### Phase 3: Missing Value Handling
- **Statistical Imputation**: 
  - Continuous variables: KNN imputation
  - Categorical variables: Mode imputation by wave
- **Temporal Consistency**: Apply domain knowledge for specific variables (e.g., age progression)
- **Data Standardization**: Normalize continuous features for model compatibility

### 2. Feature Engineering and Selection

#### Statistical Feature Selection
- **ANOVA F-Scores**: Rank features based on their discriminative power
- **K-Best Selection**: Select top K most relevant features for prediction
- **Multicollinearity Assessment**: Evaluate feature correlations to avoid redundancy

#### Feature Engineering
- **Domain-Specific Transformations**: Apply clinical and psychological domain knowledge
- **Interaction Terms**: Create meaningful combinations of variables
- **Normalization**: Standardize features for algorithm compatibility

### 3. Prediction Modeling Framework

#### Multiple Algorithm Support
The framework supports various machine learning algorithms:

1. **Random Forest**: Ensemble method robust to noise and outliers
2. **Logistic Regression**: Interpretable linear model with regularization
3. **Support Vector Machine**: Non-linear decision boundaries with kernel methods

#### Model Optimization
- **Hyperparameter Tuning**: Grid search and cross-validation for optimal parameters
- **Ensemble Methods**: Combine multiple models for improved performance
- **Regularization**: Prevent overfitting through penalty terms

### 4. Time-Aware Cross-Validation Strategies

Three distinct cross-validation schemes designed for longitudinal data:

#### Strategy 1: Fixed Split Validation
- **Training**: Early waves (e.g., Wave 1)
- **Internal Validation**: Subsequent wave (e.g., Wave 2)
- **Testing**: Later wave (e.g., Wave 3)
- **External Validation**: Final wave (e.g., Wave 4)

#### Strategy 2: Sliding Window Validation
- **Window 1**: Train on Wave 1, Test on Wave 2
- **Window 2**: Train on Waves 1-2, Test on Wave 3
- **Window 3**: Train on Waves 1-3, Test on Wave 4
- **Purpose**: Assess temporal generalization capability

#### Strategy 3: Cumulative Learning Validation
- **Round 1**: Train on Wave 1, Test on Wave 2
- **Round 2**: Train on Waves 1-2, Test on Wave 3
- **Round 3**: Train on Waves 1-3, Test on Wave 4
- **Purpose**: Simulate real-world learning scenarios

### 5. Comprehensive Model Evaluation

#### Performance Metrics
The models are evaluated using comprehensive performance metrics:

- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **PRC AUC**: Area under the Precision-Recall Curve (particularly relevant for imbalanced data)
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall (Sensitivity)**: Proportion of true positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: Proportion of true negatives correctly identified

#### Statistical Significance Testing
- **Bootstrap Confidence Intervals**: 95% confidence intervals for all metrics
- **McNemar's Test**: Compare performance between different models
- **DeLong Test**: Compare ROC curves statistically

## Key Variables Categories

### Demographic and Socioeconomic Variables
- Age, Gender, Grade Level, Body Mass Index (BMI)
- Nationality, Geographic Region
- Parental Education Levels (Mother/Father)
- Parental Occupation Categories (Mother/Father)
- Family Income, Family Structure (Only Child Status)

### Psychological and Behavioral Measures
- **CBCL Scales**: Child Behavior Checklist comprehensive assessment
- **CESD**: Center for Epidemiologic Studies Depression Scale
- **SCARED**: Screen for Child Anxiety Related Emotional Disorders
- **IAT**: Internet Addiction Test and related measures
- **CRIES-COVID**: Pandemic-related stress and coping mechanisms

### Academic and Social Indicators
- Academic performance metrics
- Social relationship quality measures
- Peer interaction assessments
- School engagement indicators

### Physical Health Metrics
- Anthropometric measurements (Height, Weight)
- Blood pressure readings (Systolic/Diastolic)
- Visual acuity assessments
- General physical health indicators

## Configuration and Customization

### Flexible Framework Design
The framework is designed to be adaptable to different cohort studies:

```python
# Example configuration for a 5-wave study
class Config:
    WAVE_PREFIX = 'T'      # Prefix for time waves
    N_WAVES = 5           # Number of time waves
    TARGET_VAR = 'NSSI'   # Outcome variable name
    N_FEATURES = 50       # Number of features to select
```

### Extensible Components
- **Custom Models**: Add new algorithms through inheritance
- **New Validators**: Implement additional cross-validation strategies
- **Alternative Metrics**: Extend evaluation with domain-specific measures
- **Preprocessing Pipelines**: Customize data transformations

## Confidentiality and Ethical Compliance

The datasets generated and/or analyzed during this study are not publicly available due to the sensitive nature of the data involving human participants. Anonymized data may be available from the corresponding author upon reasonable request.

All implementations adhere to strict ethical guidelines for working with adolescent populations and maintain full compliance with institutional review board protocols.

## Usage Instructions

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
# Initialize the predictor framework
from consolidated_nssi_prediction import LongitudinalPredictor

# Create predictor instance
predictor = LongitudinalPredictor()

# Run complete pipeline
predictor.run_complete_pipeline('path/to/your/data.csv')
```

### Custom Configuration

```python
# Customize framework parameters
class CustomConfig:
    N_FEATURES = 30
    CV_FOLDS = 10
    TARGET_VAR = 'custom_outcome'

# Use custom configuration
predictor = LongitudinalPredictor(config=CustomConfig)
```

## Results Summary

The framework demonstrates that machine learning approaches can effectively predict outcomes in longitudinal cohort studies. Key findings from validation include:

1. **Temporal Stability**: Models maintain performance across different time periods
2. **Feature Robustness**: Selected features show consistent predictive power
3. **Methodological Soundness**: Multiple validation strategies confirm reliability
4. **Clinical Relevance**: Identified risk factors align with theoretical expectations

## Citation

If you use this framework or methodology in your research, please cite:

```bibtex
@article{guo2025longitudinal,
  title={Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents: A prospective multicenter Cohort study},
  author={Guo, Xinyu and Liu, Shuyi and Jiang, Lihua and Xiong, Zhihan and Wang, Linna and Lu, Li and Li, Xiang and Zhao, Li and Shek, Daniel TL},
  journal={Journal of Affective Disorders},
  pages={120110},
  year={2025},
  publisher={Elsevier}
}
```

## Contributing

Contributions to improve the framework are welcome. Please follow standard GitHub workflows:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author Information

This framework was developed as part of a research initiative on adolescent mental health and behavioral prediction. For questions regarding the methodology or implementation, please contact the corresponding author of the associated research paper.