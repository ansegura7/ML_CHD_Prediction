####################################################################################################
# Title: Data Analysis of Patients with Risk of Presenting CHD, using Machine Learning Techniques  # 
# Created by: Segura Tinoco Andrés y Orozco Cacique Johana                                         #
# Created on: July 7, 2017                                                                         #
####################################################################################################

# Delete variables from memory
rm(list = ls(all = TRUE))

#########################################
# 1. Loading libraries and source data  #
#########################################

# Loading R libraries
suppressWarnings(library(e1071))
suppressWarnings(library(kknn))
suppressWarnings(library(MASS))
suppressWarnings(library(class))
suppressWarnings(library(rpart))
suppressWarnings(library(randomForest))
suppressWarnings(library(ada))

# Loading K-folds library
suppressWarnings(library(caret))

# Loading FactoMineR library
suppressWarnings(library(FactoMineR))

# Loading plotting libraries
suppressWarnings(library(ggplot2))

# Loading data
saha_data <- read.csv("ML_CHD_Prediction/data/SAheart.csv", header = TRUE, sep = ',', dec = '.')

#################################
# 2. Descriptive Data Analysis  #
#################################

# Show dataframe dim(rows, cols)
dim(saha_data)

# Summary of dataset
# Note: famhist -> Present = 1, Absent = 0
head(saha_data, n = 10)

# Boxplots are created to identify the outliers for each variable
# famhist and chd variables are not analyzed, because they are dichotomous variables
boxplot(x = saha_data[, c(-5, -10)], range = 2, border = c("blue", "green", "black", "orange"), las=2, cex.axis=0.9, cex.names=0.9)

# The Scatterplot is created with the correlation matrices
pairs(saha_data)

# Apply PCA
pca.res <- PCA(saha_data[, c(-5, -10)], scale.unit = TRUE, graph = FALSE)
pca.res$eig

# The Correlation circle is plotted - Only variables that have cos2 > 0.25 (25%)
plot(pca.res, axes=c(1, 2), choix="var", col.var="blue",new.plot=TRUE, select="cos2 0.25")

##############################################
# 3. The Base Reference Error is calculated  #
##############################################

# Get rows number
nRows <- nrow(saha_data)

# Calculate the amount of Yes and No for the CHD variable
nYes <- sum(saha_data$chd == "Yes")
nNo <- nRows - nYes

# The presence percentage of the Yes class is calculated
nYesClass <- nYes * 100 / nRows
cat("# Yes: ", nYes, ", # No: ", nNo, ", % Yes Class: ", nYesClass)

#############################################
# 4. Training and Calibration of the model  #
#############################################

# Start the ML process
ptm <- proc.time()

# Get CHD index
ixTypeVar = ncol(saha_data)

# Cross-Validation iters
nCV <- 10

# K-folds number
nFolds <- 10

# Variables to store the detection of CHD = Yes
detection.yes.svm <- rep(0, nCV)
detection.yes.knn <- rep(0, nCV)
detection.yes.bayes <- rep(0, nCV)
detection.yes.dtree <- rep(0, nCV)
detection.yes.forest <- rep(0, nCV)
detection.yes.adaboost <- rep(0, nCV)

# Variables to store the detection of CHD = No
detection.no.svm <- rep(0, nCV)
detection.no.knn <- rep(0, nCV)
detection.no.bayes <- rep(0, nCV)
detection.no.dtree <- rep(0, nCV)
detection.no.forest <- rep(0, nCV)
detection.no.adaboost <- rep(0, nCV)

# Variables to store the average global error
error.global.svm <- rep(0, nCV)
error.global.knn <- rep(0, nCV)
error.global.bayes <- rep(0, nCV)
error.global.dtree <- rep(0, nCV)
error.global.forest <- rep(0, nCV)
error.global.adaboost <- rep(0, nCV)

# Start cross-validation
for(i in 1:nCV) {
  
  # Groups are created (for this case 10)
  groups <- createFolds(1:nRows, k = nFolds)
  
  # Internal counters of the prediction of the Yes
  yes.svm <- 0
  yes.knn <- 0
  yes.bayes <- 0
  yes.dtree <- 0
  yes.forest <- 0
  yes.adaboost <- 0
  
  # Internal counters of the prediction of the No
  no.svm <- 0
  no.knn <- 0
  no.bayes <- 0
  no.dtree <- 0
  no.forest <- 0
  no.adaboost <- 0
  
  # Internal error counters of the models
  error.svm <- 0
  error.knn <- 0
  error.bayes <- 0
  error.dtree <- 0
  error.forest <- 0
  error.adaboost <- 0
  
  # Loop to perform the cross-validation
  for(k in 1:nFolds) {
    
    # The learning and testing datasets are generated
    sample <- groups[[k]]
    tlearning <- saha_data[-sample, ]
    ttesting <- saha_data[sample, ]
    
    # 1. A model is generated with the Vector Support Machines method
    model <- svm(chd~., data = tlearning, kernel = "linear")
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.svm <- yes.svm + MC[2, 2]
    no.svm <- no.svm + MC[1, 1]
    error.svm <- error.svm + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 2. A model is generated with the Nearest Neighbors method with K = 5
    model <- train.kknn(chd~.,data = tlearning, kmax = 5)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.knn <- yes.knn + MC[2, 2]
    no.knn <- no.knn + MC[1, 1]
    error.knn <- error.knn + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 3. A model is generated with the Naive Bayes method
    model <- naiveBayes(chd~., data = tlearning)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.bayes <- yes.bayes + MC[2, 2]
    no.bayes <- no.bayes + MC[1, 1]
    error.bayes <- error.bayes + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 4. A model is generated with the Decision Trees method
    model <- rpart(chd~., data = tlearning)
    prediction <- predict(model, ttesting, type='class')
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.dtree <- yes.dtree + MC[2, 2]
    no.dtree <- no.dtree + MC[1, 1]
    error.dtree <- error.dtree + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 5. A model is generated with the Random Forest method with 300 Trees
    model <- randomForest(chd~., data = tlearning, importance = TRUE, ntree = 300)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.forest <- yes.forest + MC[2, 2]
    no.forest <- no.forest + MC[1, 1]
    error.forest <- error.forest + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 6. A model is generated with the ADA Boosting method
    model <- ada(chd~., data = tlearning, iter = 50, nu = 1, type="real")
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    yes.adaboost <- yes.adaboost + MC[2, 2]
    no.adaboost <- no.adaboost + MC[1, 1]
    error.adaboost <- error.adaboost + (1 - (sum(diag(MC)) / sum(MC))) * 100
  }
  
  # The amount of Yes detected by each method is saved
  detection.yes.svm[i] <- yes.svm
  detection.yes.knn[i] <- yes.knn
  detection.yes.bayes[i] <- yes.bayes
  detection.yes.dtree[i] <- yes.dtree
  detection.yes.forest[i] <- yes.forest
  detection.yes.adaboost[i] <- yes.adaboost
  
  # The amount of No detected by each method is saved
  detection.no.svm[i] <- no.svm
  detection.no.knn[i] <- no.knn
  detection.no.bayes[i] <- no.bayes
  detection.no.dtree[i] <- no.dtree
  detection.no.forest[i] <- no.forest
  detection.no.adaboost[i] <- no.adaboost
  
  # The average global error is saved for each model
  error.global.svm[i] <- error.svm / nFolds
  error.global.knn[i] <- error.knn / nFolds
  error.global.bayes[i] <- error.bayes / nFolds
  error.global.dtree[i] <- error.dtree / nFolds
  error.global.forest[i] <- error.forest / nFolds
  error.global.adaboost[i] <- error.adaboost / nFolds
}

###################################
# 5. Plotting the Models Results  #
###################################

# The limits for Plot 1 are calculated
yLim <- c(min(error.global.svm, error.global.knn, error.global.bayes, error.global.dtree, error.global.forest, error.global.adaboost) * 0.9,
          max(error.global.svm, error.global.knn, error.global.bayes, error.global.dtree, error.global.forest, error.global.adaboost) * 1.3)

# The curves are plotted with the average global error of each model
plot(error.global.svm, col = "magenta", type = "b", ylim = yLim, main = "Detection of people prone to CHD",
     xlab = "Número de iteración", ylab = "Global Error", cex.axis = 0.9)
points(error.global.knn, col = "blue", type = "b")
points(error.global.bayes, col = "red", type = "b")
points(error.global.dtree, col = "lightblue3", type = "b")
points(error.global.forest, col = "olivedrab", type = "b")
points(error.global.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "DTree", "RandomF", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 2 are setted
yLim <- c(0, nYes)

# The curves of the Yes detection are plotted
plot(detection.yes.svm, col = "magenta", type = "b", ylim = yLim, main = "Detection of people who are prone to CHD",
     xlab = "Número de iteración", ylab = "Yes detected", cex.axis = 0.9)
points(detection.yes.knn, col = "blue", type = "b")
points(detection.yes.bayes, col = "red", type = "b")
points(detection.yes.dtree, col = "lightblue3", type = "b")
points(detection.yes.forest, col = "olivedrab", type = "b")
points(detection.yes.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "DTree", "RandomF", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 3 are setted
yLim <- c(0, nNo)

# The curves of the No detection are plotted
plot(detection.no.svm, col = "magenta", type = "b", ylim = yLim, main = "Detection of people who are NOT prone to CHD",
     xlab = "Número de iteración", ylab = "No detected", cex.axis = 0.9)
points(detection.no.knn, col = "blue", type = "b")
points(detection.no.bayes, col = "red", type = "b")
points(detection.no.dtree, col = "lightblue3", type = "b")
points(detection.no.forest, col = "olivedrab", type = "b")
points(detection.no.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "DTree", "RandomF", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The average Errors of the models are shown
print(mean(error.global.svm))
print(mean(error.global.knn))
print(mean(error.global.bayes))
print(mean(error.global.dtree))
print(mean(error.global.forest))
print(mean(error.global.adaboost))

# The Yes detected by each models are shown
print(mean(detection.yes.svm))
print(mean(detection.yes.knn))
print(mean(detection.yes.bayes))
print(mean(detection.yes.dtree))
print(mean(detection.yes.forest))
print(mean(detection.yes.adaboost))

# The No detected by each models are shown
print(mean(detection.no.svm))
print(mean(detection.no.knn))
print(mean(detection.no.bayes))
print(mean(detection.no.dtree))
print(mean(detection.no.forest))
print(mean(detection.no.adaboost))

# End of the ML process
ptm <- (proc.time() - ptm)
cat("Time: ", ptm)

#####################
# 6. End of Script  #
#####################