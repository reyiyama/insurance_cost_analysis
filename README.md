# Predictive Analysis of Medical Insurance Charges

## Overview

This project aims to predict medical insurance charges based on patient demographics and health indicators using statistical analysis and regression modeling techniques. The dataset used contains 2,772 records including age, sex, BMI, number of children, smoking status, region, and medical charges as the target variable. This analysis seeks to determine the significant factors contributing to the medical expenses and build a predictive model for future estimates.

The data was obtained from [Kaggle](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction/data), and the analysis was conducted using R. The results and detailed analysis, including diagnostic checks and model adequacy, are presented in this repository.

### Dataset Overview
- **Dataset Source**: [Kaggle - Medical Insurance Cost Prediction](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction/data)
- **Number of Rows**: 2,772
- **Variables**:
  - **Age**: Age of the individual.
  - **Sex**: Gender of the individual (male or female).
  - **BMI**: Body Mass Index.
  - **Children**: Number of children.
  - **Smoker**: Smoking status (yes or no).
  - **Region**: Geographical region.
  - **Charges**: Medical charges billed by health insurance.

### Objectives
1. Identify the primary factors influencing medical expenses.
2. Assess the accuracy of regression models in predicting insurance charges.
3. Improve decision-making for health insurance pricing and risk assessment.

## Analysis Steps

### 1. Data Preparation and Exploration
- **Data Loading**: The dataset was loaded using the `read_csv()` function in R, followed by checking for missing values. No missing values were found in the dataset.
- **Exploratory Data Analysis (EDA)**: Summary statistics and visualizations were used to understand the relationships between variables.
  - **Summary Statistics**: Descriptive statistics for each feature were computed, providing insight into the distribution, central tendency, and spread of each variable.
  - **Pairwise Plot Analysis**: A pairwise plot (`ggpairs`) was generated to observe correlations between predictors and the target variable, highlighting significant relationships like age, BMI, and smoker status with medical charges.
  - **Histograms**: A histogram for medical charges showed a right-skewed distribution, indicating a need for transformation.
  - **Boxplots**: Boxplots were used to visualize the distribution of charges across different categories such as smoking status, sex, and region, revealing disparities in medical charges among different groups.

### 2. Data Transformation
- **Box-Cox Transformation**: A Box-Cox transformation was applied to stabilize the variance. The optimal lambda value suggested a logarithmic transformation, which was subsequently applied to the charges variable to address skewness and heteroscedasticity.
- **Data Splitting**: The dataset was split into training (70%) and testing (30%) sets to validate model performance. A random seed was set to ensure reproducibility of results.

### 3. Regression Model Building
- **Multiple Linear Regression**: A linear regression model was built using the log-transformed charges variable.
  - **Model Fitting**: The regression model used `age`, `sex`, `BMI`, `children`, `smoker`, and `region` as predictors. These variables were chosen based on their relevance to medical expenses and their significance in the exploratory analysis.
  - **Categorical Variables Encoding**: Categorical variables (`sex`, `smoker`, `region`) were converted to factors to be used in the regression model.
  - **Model Summary**: The model explained approximately 76.8% of the variance in charges, with predictors such as smoking status, age, and BMI being highly significant.
  - **Interpretation**: Smoking status was found to have the largest impact on medical charges, significantly increasing the predicted cost. Age and BMI were also positively associated with higher charges, while sex and region had smaller but notable effects.

### 4. Model Diagnostics
- **Model Adequacy Check**:
  - **Residual Plots**: Residual vs. Fitted, Q-Q plot, Scale-Location, and Leverage plots were used to assess model assumptions. The Residual vs. Fitted plot showed a non-random pattern, indicating potential issues with non-linearity or heteroscedasticity.
  - **Durbin-Watson Test**: The Durbin-Watson statistic was close to 2, indicating no significant autocorrelation in the residuals.
  - **Shapiro-Wilk Test**: Residuals deviated from normality (p-value < 2.2e-16), suggesting that the normality assumption was violated.
  - **Breusch-Pagan Test**: Significant heteroscedasticity was found (p-value < 2.2e-16), indicating non-constant variance in the residuals.
  - **Variance Inflation Factor (VIF)**: Multicollinearity was checked using VIF, and all values were below 2, indicating no significant multicollinearity among the predictors.

### 5. Outlier and Influential Points Analysis
- **Outlier Tests**: Outliers were identified using studentized residuals and Bonferroni-adjusted p-values. Several observations were found to have significant influence on the model.
- **Cook's Distance**: Influential points were flagged using Cook's distance. Observations with high Cook's distance values were analyzed further to assess their impact on model stability.
- **Leverage and Influence Measures**: Additional influence measures such as DFBETAS and leverage values were used to identify data points with a disproportionate impact on the model coefficients.

### 6. Model Improvement
- **Corrective Methods**:
  - **Data Cleaning**: Influential points were removed to improve model stability. A new regression model was fit using the cleaned dataset, which showed improved diagnostic metrics.
  - **Variable Selection**: Subset selection and stepwise regression (both forward and backward) were used to identify the best subset of predictors. The most predictive model included `age`, `BMI`, `children`, and `smoker status`.
  - **Model Comparison**: The original model, which included the `region` variable, was compared with the refined model using ANOVA and PRESS values. Despite minor improvements in MSE for the refined model, the original model with `region` was found to be a better fit due to a significantly better F-statistic and lower PRESS value.

### 7. Final Model Assessment
- **Final Model Summary**: The final model included `age`, `sex`, `BMI`, `children`, `smoker`, and `region`. Smoking status remained the most significant predictor of medical charges, with a large positive effect. Age, BMI, and the number of children also contributed positively, while certain regions showed lower predicted charges compared to others.
- **Model Diagnostics for Final Model**: The final model was subjected to a comprehensive diagnostic check. The residual plots indicated slight non-linearity and heteroscedasticity. The Shapiro-Wilk and Breusch-Pagan tests confirmed issues with normality and constant variance, respectively. However, the model still provided a good fit with an adjusted R-squared of 76.39%.

## Code Explanation

The R script (`Regression_final_assignment_git.Rmd`) contains all the steps performed during the analysis, which include:

### 1. Data Loading and Preprocessing
- **Loading the Dataset**: The dataset was loaded using `read_csv()` from the `tidyverse` package.
- **Data Cleaning**: The dataset was checked for missing values (`sum(is.na())`) and data types were verified.
- **Categorical Encoding**: Categorical variables like `sex`, `smoker`, and `region` were converted to factors to allow proper modeling.

### 2. Exploratory Data Analysis (EDA)
- **Summary Statistics**: Functions such as `summary()` were used to compute basic statistics.
- **Visualization**: The `GGally` package was used for generating pairwise plots (`ggpairs()`), and `ggplot2` was used to create histograms and boxplots for visualizing distributions.
- **Correlation Analysis**: Correlations between features and medical charges were visualized to identify significant relationships.

### 3. Data Transformation
- **Box-Cox Transformation**: The `boxcox()` function from the `MASS` package was used to determine the appropriate transformation for stabilizing variance. A logarithmic transformation was applied to the `charges` variable.
- **Splitting the Data**: The dataset was split into training and testing sets using `sample()` and indexing techniques.

### 4. Model Building
- **Multiple Linear Regression**: The `lm()` function was used to build the regression model with `log(charges)` as the dependent variable and `age`, `sex`, `BMI`, `children`, `smoker`, and `region` as predictors.
- **Model Summary**: The `summary()` function provided details on coefficients, R-squared, and significance levels for each predictor.

### 5. Model Diagnostics
- **Residual Analysis**: The diagnostic plots (`plot()`) included Residual vs. Fitted, Q-Q plots, and Scale-Location plots.
- **Durbin-Watson Test**: The `durbinWatsonTest()` function from the `car` package was used to check for autocorrelation in the residuals. The test result indicated no significant autocorrelation.
- **Normality Check**: The `shapiro.test()` function was used to perform the Shapiro-Wilk test on the residuals. The result indicated that the residuals were not normally distributed.
- **Breusch-Pagan Test**: The `bptest()` function from the `lmtest` package was used to check for heteroscedasticity. The test indicated significant heteroscedasticity, suggesting non-constant variance in the residuals.

### 6. Outlier and Influential Points Analysis
- **Studentized Residuals**: Studentized residuals were calculated to identify potential outliers. Observations with high studentized residuals were flagged for further investigation.
- **Cook's Distance**: The `cooks.distance()` function was used to calculate Cook's distance for each observation. Points with high Cook's distance values were identified as influential and were analyzed further.
- **Leverage Values**: Leverage values were calculated using the `hatvalues()` function to identify points that had a high influence on the regression model. Observations with high leverage were further analyzed to determine their impact on the model.

### 7. Model Refinement and Improvement
- **Data Cleaning**: Influential observations identified through Cook's distance and leverage analysis were removed, and the model was re-fit using the cleaned dataset.
- **Stepwise Regression**: The `stepAIC()` function from the `MASS` package was used for both forward and backward stepwise regression to select the best subset of predictors.
- **ANOVA Comparison**: The `anova()` function was used to compare the original model with the refined model. The comparison showed that the original model, which included the `region` variable, provided a better fit overall.

### 8. Final Model Assessment
- **Adjusted R-Squared**: The final model's adjusted R-squared value was calculated to evaluate the overall fit. The adjusted R-squared was 76.39%, indicating that the model explained a substantial portion of the variance in medical charges.
- **Prediction Accuracy**: The model's performance was evaluated on the test set by calculating the Mean Squared Error (MSE) and Mean Absolute Error (MAE). The results indicated that the model had a good predictive accuracy, although some deviations were noted for high-charge observations.

## How to Recreate This Analysis
To recreate the findings presented in this analysis, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone www.github.com/reyiyama
   ```
2. **Install Necessary Packages**:
   - Install the required R packages such as `tidyverse`, `GGally`, `MASS`, `car`, and `DAAG`.
   ```r
   install.packages(c("tidyverse", "GGally", "MASS", "car", "DAAG", "lmtest"))
   ```
3. **Run the Analysis**:
   - Load the dataset using the given R script.
   - Execute the exploratory analysis, model fitting, and diagnostics by running the provided R script (`Regression_final_assignment_git.Rmd`).

4. **Data Source**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction/data).

## Results Summary
- **Key Findings**: Smoking status is the most significant predictor of higher medical charges, followed by age, BMI, and the number of children. Regional differences also play a role, with some regions showing lower average charges.
- **Model Accuracy**: The final model explained 76.8% of the variance in medical charges. However, issues like non-normality and heteroscedasticity were noted, which were addressed through transformations and model refinements. Despite these issues, the model was found to be robust and provided meaningful insights into the factors affecting medical charges.

## Future Work
- **Feature Engineering**: Consider adding additional features such as physical activity, dietary habits, or health history to improve model accuracy and capture more aspects influencing medical expenses.
- **Machine Learning Models**: Future analysis could involve more sophisticated machine learning models such as Random Forest or Gradient Boosting to improve prediction accuracy and capture non-linear relationships.
- **Addressing Heteroscedasticity**: Further research could focus on advanced techniques to handle heteroscedasticity, such as weighted least squares regression or generalized linear models.

## Authors and Contributions
- **Saurabh Tyagi**: Data collection, exploratory data analysis, model building, and initial reporting.
- **Adyasaa Mohapatra**: Model diagnostics, adequacy checks, and handling of influential points.
- **Amay Iyer**: Model refinement, stepwise regression, documentation, and final report writing.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Kaggle** for providing the medical insurance dataset.
- **RMIT University** for guidance and support throughout this project.

## Contact
If you have any questions or feedback, feel free to reach out to the authors via GitHub.
@Sam7135 - Saurabh Tyagi
@reyiyama - Amay Iyer

---
