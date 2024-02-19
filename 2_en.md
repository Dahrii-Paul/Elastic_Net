# glmnet with 0.001 step alpha
```{R}
getwd()
setwd("~/Desktop/ml1/1_data/1_main/glmnet")
model_main <- readRDS("model.rds")
library(glmnet)
library(dplyr)

df <- read.csv("~/Desktop/ml1/1_data/1_main/trainMain_norm.csv")
df2 <- read.csv("~/Desktop/ml1/1_data/1_main/testMain_norm.csv")
dim(df)
x.train <- as.matrix(df[,1:9637])
y.train <- df[,9638]
x.test <- as.matrix(df2[,1:9637])
y.test <- df2[,9638]

# extract in a data frame format
results <- data.frame()
for (i in 0:1000) {
  fit.name <- paste0("alpha", i/1000)
  # Extracting lambda and measure information from the fitted models
  lambda <- model_main[[fit.name]]$lambda
  measure <- model_main[[fit.name]]$cvm
  temp <- data.frame(alpha=i/1000, measure=measure, lambda=lambda, fit.name=fit.name)
  results <- rbind(results, temp)
}

# Extact the min classification error and min lambda
min_measure <- min(results$measure)
rows_with_min_measure <- results[results$measure == min_measure, ]
min_lambda <- min(rows_with_min_measure$lambda)
row_with_min_lambda <- rows_with_min_measure[rows_with_min_measure$lambda == min_lambda, ]
row_with_min_lambda

# Model plot
model_main$alpha0.834
model_main$alpha0.834$glmnet.fit
plot(model_main$alpha0.834)
plot(model_main$alpha0.834$glmnet.fit)
plot_glmnet(model_main$alpha0.834$glmnet.fit, xvar = "rlambda", label = 20)
plot_glmnet(model_main$alpha0.834$glmnet.fit, xvar = "dev", label = 20)
## 11. Visualized variable importance
plot(varImp(model_main$alpha0.834$glmnet.fit))
vip(model_main$alpha0.834, method = "model")
vip(model_main$alpha0.834,num_features = 80,
    geom = "col") + theme_bw()

plot(model_main$alpha0.834$lambda, log="x",model_main$alpha0.834$cvm,xlab="Log Lambda",ylab="Classification Error")
abline(v=model_main$alpha0.834$lambda.min, lty=2, col="red")
abline(v=model_main$alpha0.834$lambda.1se, lty=2, col="green")
legend("topright", legend = c("lambda.min = 0.106", "lambda.1se = 0.1101"), lty = 2, col = c("red", "green"))
#title(main = "Plot of Classification Error vs. Log Lambda")


plot(log(model_main$alpha0.834$lambda), model_main$alpha0.834$cvm, pch = 19, col = "red",
     xlab = "log(Lambda)", ylab = model_main$alpha0.834$name)
points(log(model_main$alpha0.001$lambda), model_main$alpha0.001$cvm, pch = 19, col = "green")
points(log(model_main$alpha1$lambda), model_main$alpha1$cvm, pch = 19, col = "blue")
legend("topleft", legend = c("alpha = 0.834", "alpha = 0.001", "alpha = 1"),
       pch = 19, col = c("red", "green", "blue"))

#-----------------------------------------------------#
# Get coefficient
coef_seq <-coef(model_main$alpha0.834$glmnet.fit, s = model_main$alpha0.834$lambda.1se)  %>% as.matrix()
# Get the features whose coef is not equal to zeros
feat_seq <- rownames(coef_seq)[-1][which(coef_seq[-1, 1] != 0)]
length(feat_seq)
# make coef matrix to data frame
coef_seq1 <- as.data.frame(coef_seq)  %>% slice(-1) %>% filter(s1 !=0)
# rank the order
coef_seq1 <-coef_seq1 %>% arrange(desc(abs(s1)))
coef_seq1$rank <- 1:nrow(coef_seq1)
write.csv(coef_seq1,"coef_feature.csv")
# selected feature
selected_features_train <- x.train[, feat_seq, drop = FALSE]
selected_features_test <- x.test[, feat_seq, drop = FALSE]
FSelastic_train <- cbind(selected_features_train, y.train)
colnames(FSelastic_train)[81] <- "target"
FSelastic_test <- cbind(selected_features_test, y.test)
colnames(FSelastic_test)[81] <- "target"
write.csv(FSelastic_train, "FS_train.csv", row.names = FALSE)
write.csv(FSelastic_test, "FS_test.csv", row.names = FALSE)
## 13. Visualize the coef of the models (positive & negative coef)
coef_seq1$Coefficient <- ifelse(coef_seq1$s1 >= 0, "Positive", "Negative")
#plot positive and negative
ggplot(coef_seq1, aes(x = s1, y = reorder(rownames(coef_seq1), s1), fill = Coefficient)) +
  geom_bar(stat = "identity") +
  labs(title = "Coefficient in Elastic Model",
       x = "Coefficient",
       y = "Features") +
  scale_fill_manual(values = c("Negative" = "red", "Positive" = "blue")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

library(coefplot)
coefpath(model_main$alpha0.834$glmnet.fit)
coefpath(model_main$alpha0.834$glmnet.fit, labelMin="lambda_min",colorMin = "black",)
coefplot(model_main$alpha0.834$glmnet.fit, lambda=model_main$alpha0.834$lambda.min, sort = "magnitude")
#------------------------------------------------------------#
# build tune model
# load library
library(glmnet)
library(dplyr)
library(doParallel)
library(parallel)
library(plotmo)
library(vip)
library(caret)
library(mltools)

# 1. Model created tuned
elnet.model <- glmnet(x.train, y.train,
                      alpha = row_with_min_lambda$alpha, 
                      lambda = model_main$alpha0.843$lambda.1se,
                      family = "binomial")
coef(elnet.model)

# 2. train
observed.classes <- df$label
probabilities <- elnet.model %>% predict(newx = x.train,  type="response")
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
acc_train = mean(predicted.classes == observed.classes) # Model accuracy = 0.8886128
conf_matrix_train <- table(observed.classes, predicted.classes)
# Sensitivity and Specificity for Training Set
sensitivity_train <- conf_matrix_train[2, 2] / sum(conf_matrix_train[2, ])
specificity_train <- conf_matrix_train[1, 1] / sum(conf_matrix_train[1, ])
# AUC for Training Set
roc_train <- pROC::roc(as.numeric(observed.classes), as.numeric(probabilities))
auc_train <- pROC::auc(roc_train)

# Calculate MCC for Training Set
TP <- as.numeric(conf_matrix_train[2, 2])
FP <- as.numeric(conf_matrix_train[1, 2])
TN <- as.numeric(conf_matrix_train[1, 1])
FN <- as.numeric(conf_matrix_train[2, 1])
mcc_train <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
sensitivity_train1 <- TP / (TP + FN)
specificity_train1 <- TN / (TN + FP)
acc <- (TP+TN)/(TP+TN+FP+FN)
acc1 <- mean(predicted.classes == observed.classes)

# 3. test
observed.classes <- df2$label
prob.elnet <- elnet.model%>% predict(newx = x.test,  type="response")
predicted.classes <- ifelse(prob.elnet > 0.5, "1", "0")
acc_test = mean(predicted.classes == observed.classes)               # Model accuracy = 0.8640133
conf_matrix_test <-  table(observed.classes, predicted.classes) #Confusion matrix
# Sensitivity and Specificity for Training Set
sensitivity_test <- conf_matrix_test[2, 2] / sum(conf_matrix_test[2, ])
specificity_test <- conf_matrix_test[1, 1] / sum(conf_matrix_test[1, ])
# AUC for Training Set
roc_test <- pROC::roc(as.numeric(observed.classes), as.numeric(prob.elnet))
auc_test <- pROC::auc(roc_test)
# Calculate MCC for Training Set
TP <- as.numeric(conf_matrix_test[2, 2])
FP <- as.numeric(conf_matrix_test[1, 2])
TN <- as.numeric(conf_matrix_test[1, 1])
FN <- as.numeric(conf_matrix_test[2, 1])
mcc_test <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# result
results_df <- data.frame(
  Metric = c("Sensitivity", "Specificity", "AUC", "MCC", "Accuracy"),
  Train = c(sensitivity_train, specificity_train, auc_train, mcc_train, acc_train),
  Test = c(sensitivity_test, specificity_test, auc_test, mcc_test, acc_test)
)
write.csv(results_df,"results_elst.csv")

#------------------------------------------------------------------------------#
############################ Feature Selected 80 ###############################
#------------------------------------------------------------------------------#
fs_train <- read.csv("FS_train.csv")
fs_test <- read.csv("FS_test.csv")

x.trainFS <- as.matrix(fs_train[,1:80])
y.trainFS <- fs_train[,81]
x.testFS <- as.matrix(fs_test[,1:80])
y.testFS <- fs_test[,81]

# 1. Model created tuned

elnet.model <- glmnet(x.trainFS, y.trainFS,
                      alpha = row_with_min_lambda$alpha, 
                      lambda = model_main$alpha0.843$lambda.1se,
                      family = "binomial")

# select the top 10 features
top_10_features <- rownames(coef_seq1)[][which(coef_seq1[, 2] <= 10)]
selected_features_train_10 <- x.train[, top_10_features, drop = FALSE]
selected_features_test_10 <- x.test[, top_10_features, drop = FALSE]
FS_train10 <- cbind(selected_features_train_10, y.train)
colnames(FS_train10)[11] <- "target"
FS_test10 <- cbind(selected_features_test_10, y.test)
colnames(FS_test10)[11] <- "target"
write.csv(FS_train10, "FS_train10.csv", row.names = FALSE)
write.csv(FS_test10, "FS_test10.csv", row.names = FALSE)

```
