# glmnet Model
```R
getwd()
setwd("/Users/paul/Desktop/ml1/1_data/1_main")
list.files()

# libraries
# load library
library(glmnet)
library(dplyr)
library(doParallel)
library(parallel)
library(plotmo)
library(vip)
library(caret)
library(doMC)

# load data
df <- read.csv("trainMain_norm.csv")
df2 <- read.csv("testMain_norm.csv")
dim(df)
x.train <- as.matrix(df[,1:9637])
y.train <- df[,9638]
x.test <- as.matrix(df2[,1:9637])
y.test <- df2[,9638]

cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
cl

list.of.fits <- list()
set.seed(123)
for (i in 0:1000) {
  fit.name <- paste0("alpha", i/1000)
  
  list.of.fits[[fit.name]] <-
    cv.glmnet(x.train, y.train, type.measure="class", alpha=i/1000, 
              family="binomial", parallel = TRUE)
}

results <- data.frame()
for (i in 0:1000) {
  fit.name <- paste0("alpha", i/1000)
  
  # Extracting lambda and measure information from the fitted models
  lambda <- list.of.fits[[fit.name]]$lambda
  measure <- list.of.fits[[fit.name]]$cvm
  
  temp <- data.frame(alpha=i/1000, measure=measure, lambda=lambda, fit.name=fit.name)
  results <- rbind(results, temp)
}
#View(results)
table(results$fit.name)
saveRDS(list.of.fits, "model.rds")
write.csv(results,"results.csv")



```
