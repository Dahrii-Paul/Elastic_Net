# Caret elastic net
```R
getwd()
setwd('/Users/paul/Desktop/ml1/1_data/')
list.files()

# 1. libraries
library(caret)
library(dplyr)
library(glmnet)
library(doParallel)
library(parallel)
library(plotmo)

# 2. Data Main
trainMain <- read.csv("1_main/trainMain.csv")
trainMain$chroY <- NULL
trainMain$MWW <- NULL
testMain <- read.csv("1_main/testMain.csv")
testMain$chroY <- NULL
testMain$MWW <- NULL
x.train <- as.matrix(trainMain[,1:9635])
y.train <- trainMain[,9636]
x.test <- as.matrix(testMain[,1:9635])
y.test <- testMain[,9636]

# 3. Set Seeds & register cluster
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
subsetSizes <- c(1:9635)
set.seed(123)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(10000, length(subsetSizes) + 1)
seeds[[51]] <- sample.int(10000, 1)
y.train <- as.factor(y.train)
caret_tune <- caret::train(x.train,
                           y.train,
                           method="glmnet",
                           family = "binomial",
                           preProc = c("range"),
                           parallel=TRUE,
                           metric = "Accuracy",
                           trControl = trainControl(method = "repeatedcv",
                                                    number=10,
                                                    repeats = 5),
                           tuneLength=100)
saveRDS(caret_tune, "caret_tune.rds")
#my_model <- readRDS("model.rds")
caret_tunegrid <- train(x.train,
                       y.train,
                       method="glmnet",
                       family = "binomial",
                       preProc = c("range"),
                       metric = "Accuracy",
                       trControl = trainControl(method = "repeatedcv",
                                                number=10,
                                                repeats = 5),
                       parallel=TRUE,
                       tuneGrid = expand.grid(
                         alpha = seq(0, 1,length=200),
                         lambda = seq(0.00001, 0.01, length=100)))
saveRDS(caret_tunegrid, "caret_tunegrid.rds")
stopCluster(cl)
plot(caret_tunegrid)






## 2. Check for the bestTune
get_best_result = function(caret_tunegrid) {
  best = which(rownames(caret_tunegrid$results) == rownames(caret_tunegrid$bestTune))
  best_result = caret_tunegrid$results[best, ]
  rownames(best_result) = NULL
  best_result}
model_tunegrid <-get_best_result(caret_tunegrid)
caret_tunegrid$bestTune
model_tunegrid$model_name <- "TuneGrid"

## 10. Visualized only lambda & dev
plot(caret_tunegrid$finalModel)
plot_glmnet(caret_tunegrid$finalModel, xvar = "rlambda", label = 10)
plot_glmnet(caret_tunegrid$finalModel, xvar = "dev", label = 10)

# glmnet



```
