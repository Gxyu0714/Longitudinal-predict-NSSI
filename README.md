# Longitudinal Prediction of Non-Suicidal Self-Injury (NSSI) Among Chinese Adolescents

## Project Overview

This project focuses on predicting non-suicidal self-injury (NSSI) among Chinese adolescents using longitudinal machine learning approaches. The study utilizes a prospective multicenter cohort design to analyze NSSI patterns and develop predictive models.

Based on the research paper: "Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents: A prospective multicenter Cohort study"

## Repository Structure

```
.
├── consolidated_nssi_prediction.py  # Complete workflow in a single file
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents_A prospective multicenter Cohort study.pdf
```

## Methodology

### 1. Data Processing Pipeline

#### Phase 1: Data Transformation
- **Format Conversion**: Transform wide-format data to long-format for longitudinal analysis
- **Variable Organization**: Restructure 90+ variables across 4 time waves (T1, T2, T3, T4)
- **Data Validation**: Check data integrity and consistency

#### Phase 2: Data Cleaning
- **NSSI Processing**: Convert DSH (Direct Self-harm) scores to binary NSSI variable (0/1)
- **Grade Filtering**: Remove students in grades 1 and 2 from the first wave
- **Missing Value Assessment**: Comprehensive analysis of missing data patterns

#### Phase 3: Missing Value Handling
- **Temporal Imputation**: 
  - Age and Grade: Apply temporal rules (T1=T2, T3=T1+1, T4=T1+2)
  - Other continuous: Forward fill followed by backward fill
- **Advanced Imputation**:
  - Continuous variables: KNN imputation
  - Categorical variables: Mode imputation by wave

### 2. Prediction Modeling Pipeline

#### Feature Selection
- **LSTM Temporal Analysis**: Use Long Short-Term Memory networks to identify temporally relevant features
- **Feature Importance**: Rank variables based on their predictive power over time

#### Model Development
- **Algorithm**: Random Forest classifier optimized for longitudinal data
- **Temporal Considerations**: Account for time-dependent patterns in behavior

#### Validation Strategy
Three distinct cross-validation schemes to assess model robustness:

1. **Fixed Split**:
   - Training: Wave 1
   - Validation: Wave 2
   - Testing: Wave 3
   - External Validation: Wave 4

2. **Sliding Window**:
   - Window 1: Train on Wave 1, Test on Wave 2
   - Window 2: Train on Waves 1-2, Test on Wave 3
   - Window 3: Train on Waves 1-3, Test on Wave 4

3. **Cumulative Learning**:
   - Round 1: Train on Wave 1, Test on Wave 2
   - Round 2: Train on Waves 1-2, Test on Wave 3
   - Round 3: Train on Waves 1-3, Test on Wave 4

### 3. Evaluation Metrics

The models are evaluated using comprehensive performance metrics:

- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **PRC AUC**: Area under the Precision-Recall Curve
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall (Sensitivity)**: Proportion of true positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall

## Key Variables

### Demographic Variables
- Age, Gender, Grade, BMI
- Nationality, Region
- Parental education levels (mother/father)
- Parental occupation categories (mother/father)
- Family structure (only child status)

### Psychological Measures
- **CBCL Scales**: Child Behavior Checklist subscales
- **CESD**: Center for Epidemiologic Studies Depression Scale
- **SCARED**: Screen for Child Anxiety Related Emotional Disorders
- **IAT**: Internet Addiction Test
- **CRIES-COVID**: Pandemic-related stress measure

### Behavioral Indicators
- Academic performance indicators
- Social relationship measures
- Physical health metrics
- Various psychosocial factors

## Confidentiality Notice

The datasets generated and/or analyzed during this study are not publicly available due to the sensitive nature of the data involving human participants. Anonymized data may be available from the corresponding author upon reasonable request.

## Usage Instructions

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Analysis

```python
# Run the complete workflow
python consolidated_nssi_prediction.py
```

## Results Summary

The study demonstrates that machine learning approaches can effectively predict NSSI in Chinese adolescents using longitudinal data. Key findings include:

1. **Temporal Patterns**: NSSI risk factors evolve over time, requiring dynamic models
2. **Feature Importance**: Certain psychological and family factors consistently predict NSSI risk
3. **Model Performance**: The combination of LSTM feature selection and Random Forest achieves robust predictive performance
4. **Validation Robustness**: Consistent performance across multiple cross-validation schemes

## Ethical Considerations

This research adheres to strict ethical guidelines for working with adolescent populations. All data has been appropriately anonymized and handled according to institutional review board protocols.

## Author Information

This project was developed as part of a research study on adolescent mental health and behavioral prediction. For questions regarding the methodology or findings, please contact the corresponding author of the associated research paper.