# Machine Learning - CHD Risk Prediction with R
Data science project to create a predictor of CHD, based on supervised learning.

<a href="https://github.com/ansegura7/ML_CHD_Prediction/blob/master/paper/CHD_Prediction_using_ML_techniques.pdf" target="_blank">Paper</a>

## Data
The data used is a sample of 462 records of a larger dataset, described in Rousseauw et al, 1983, South African Medical Journal, belonging to a non-profit organization called South African Heart Association (SAHA).

| # | Variable  | Type  |  Description |
|---|---|---|---|
| 1 | sbp | Input | Systolic Blood Pressure |
| 2 | tobacco | Input | Cumulative Tobacco (kg) |
| 3 | ldl | Input | Low Density Lipoprotein Cholesterol |
| 4 | adiposity | Input | Adiposity |
| 5 | famhist | Input | Family history of heart disease (Present, Absent) |
| 6 | typea | Input | Type-A behavior |
| 7 | obesity | Input | Obesity |
| 8 | alcohol | Input | Current alcohol consumption |
| 9 | age | Input | Age at onset |
| 10 | chd | Target | Coronary heart disease (Yes, No) |

## Technologies and Techniques
- R 3.5.1 x64
- RStudio - Version 1.1.383
- Descriptive Data Analysis
- Supervised Learning (SL)

## R Dependencies
```{r }
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

If you need to install a package, use the following command in the R console
```{r }
  install.packages("package-name", dependencies=TRUE)
```

## Contributing and Feedback
Any kind of feedback/criticism would be greatly appreciated (algorithm design, documentation, improvement ideas, spelling mistakes, etc...).

## Authors
- Created by Andrés Segura Tinoco and Johana Orozco Cacique
- Created on July 7, 2017

## License
This project is licensed under the terms of the MIT license.
