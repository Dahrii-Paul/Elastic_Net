# Evaluate
```R
library(mlbench) #data
library(dplyr)
library(glmnet)
library(caret)
library(pROC)
library(ROCR)
library(MLmetrics)
library(mlr)

data(BreastCancer)
table(BreastCancer$Class)
BreastCancer$Id <- NULL
anyNA(BreastCancer)
BreastCancer <- na.omit(BreastCancer)
anyNA(BreastCancer)

set.seed(123) # For reproducibility
trainIndex <- createDataPartition(BreastCancer$Class, p = 0.8, list = FALSE)
trainSet <- BreastCancer[trainIndex, ]
validationSet <- BreastCancer[-trainIndex, ]
x.train <- as.matrix(trainSet[,1:9])
y.train <- trainSet[,10]
x.test <- as.matrix(trainSet[,1:9])
y.test <- trainSet[,10]

#---------------------------------------------------------------------------#
# Tunning
#---------------------------------------------------------------------------#
customSummary <- function(data, lev = NULL, model = NULL) {
  conf_matrix <- confusionMatrix(data$pred, data$obs)
  accuracy <- conf_matrix$overall["Accuracy"]
  other_metrics <- twoClassSummary(data, lev = lev, model = model)
  other_metrics1 <- prSummary(data, lev = lev, model = model)
  MCC1 <- mcc(data$pred, data$obs)
  # Extract TP, TN, FP, FN from the confusion matrix
  TP <- conf_matrix$table[1, 1]
  TN <- conf_matrix$table[2, 2]
  FP <- conf_matrix$table[2, 1]
  FN <- conf_matrix$table[1, 2]
  # Calculate MCC
  mcc <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  out <- c(other_metrics,
           other_metrics1,
           accuracy = accuracy,
           mcc = mcc,
           MCC1 = MCC1)
  out
}
train_control <- trainControl(method = "cv", 
                              number = 10,
                              savePredictions = TRUE,
                              classProbs = TRUE,
                              verboseIter = TRUE,
                              summaryFunction = customSummary)

set.seed(123)
elastic_net_model1 <- train(Class ~ .,
                            data = trainSet,
                            method = "glmnet",
                            trControl = train_control,
                            tunelength = 100,
                            metric = "ROC",
                            family = "binomial")
View(elastic_net_model1$results)



```
