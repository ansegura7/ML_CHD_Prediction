# Machine Learning - Prediction of CHD Risk with R
Data science project in R to create a predictive model of the CHD risk, based on supervised learning.

<a href="https://github.com/ansegura7/ML_CHD_Prediction/blob/master/paper/CHD_Prediction_using_ML_techniques.pdf" target="_blank">Paper</a> | <a href="https://ansegura7.github.io/ML_CHD_Prediction/code/CHD_Prediction_using_ML.html" target="_blank">Analysis</a> | <a href="https://github.com/ansegura7/ML_CHD_Prediction" target="_blank">Code</a>

## Abstract
In the current era in which we live, there is a clear and irreversible tendency to generate and store large volumes of information, from various sources such as: government agencies, public and private companies, clinics and hospitals, social networks, etc. Hence the great need to analyze the data in order to obtain some benefit for their owner, a third party or humanity in general. With this in mind, we conducted a descriptive and predictive analysis of public medical data of South Africa on patients with possible risk of presenting coronary heart disease (CHD), and applying advanced techniques of supervised machine learning and models calibration, we were able to determine when a person has high probabilities (close to 70%) of presenting or developing this disease, with the objective of being able to contribute to an early detection and diagnosis of it, for further treatment. Hopeful and convincing results were obtained, which can be improved if there is a greater amount of source data from which to learn.

## Data
The dataset used is a sample of 462 records of a larger dataset, described in Rousseauw et al, 1983, South African Medical Journal, belonging to a non-profit organization called South African Heart Association (SAHA).

| # | Variable | Type | Description | Data Type |
|---|---|---|---|---|
| 1 | sbp | Input | Systolic Blood Pressure | Numerical |
| 2 | tobacco | Input | Cumulative Tobacco (kg) | Numerical |
| 3 | ldl | Input | Low Density Lipoprotein Cholesterol | Numerical |
| 4 | adiposity | Input | Adiposity | Numerical |
| 5 | famhist | Input | Family history of heart disease (Present, Absent) | Categorical |
| 6 | typea | Input | Type-A behavior | Numerical |
| 7 | obesity | Input | Obesity | Numerical |
| 8 | alcohol | Input | Current alcohol consumption | Numerical |
| 9 | age | Input | Age at onset | Numerical |
| 10 | chd | Target | Coronary heart disease (Yes, No) | Categorical |

## Results

Based on the following results, the Naive Bayes model was selected as the best predictor for the problem we are solving, because it has an excellent Global Average Error of 28.77% and is the one with the highest YES detection index (key indicator), with 61.88%

| Model | % Global Error | % Yes Detected |  % No Detected |
|---|---|---|---|
| SVM | 28.71 | 50.00 | 82.45 |
| **Naive Bayes** | 28.77 | 61.88 | 76.16 |
| Decision Tree | 31.22 | 48.75 | 79.47 |
| Random Forest | 31.77 | 41.88 | 82.12 |
| K-NN | 33.08 | 41.88 | 80.46 |
| Base Line | 34.63 | NA | NA |
| Ada Boost | 34.95 | 43.13 | 76.82 |

## Technologies and Techniques
- R 3.5.1 x64
- RStudio - Version 1.1.383
- Descriptive Data Analysis
- Supervised Learning (SL)

## R Dependencies

``` r
  library(e1071)
  library(kknn)
  library(MASS)
  library(class)
  library(rpart)
  library(randomForest)
  library(ada)
  library(caret)
  library(FactoMineR)
```

If you need to install a package, use the following command in the R console:
``` r
  install.packages("package-name", dependencies=TRUE)
```

## Contributing and Feedback
Any kind of feedback/criticism would be greatly appreciated (algorithm design, documentation, improvement ideas, spelling mistakes, etc...).

## Authors
- Created by Segura Tinoco, Andrés and <a href="https://github.com/JohannaOrozco" target="_blank">Orozco Cacique, Johana</a>
- Created on July 7, 2017

## License
This project is licensed under the terms of the MIT license.
