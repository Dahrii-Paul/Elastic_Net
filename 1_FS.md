## Elastic Net using 
> library(caret) </br>
> library(glmnet)
>  
> **links**:
> 1.  [StatQues](https://github.com/StatQuest/ridge_lasso_elastic_net_demo/blob/master/ridge_lass_elastic_net_demo.R)
> 2.  [Manuscript on Elastic-Net Regression](https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf)
> 3.  [North Carolina State University](https://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html)
> 4.  [Kaggle1](https://www.kaggle.com/code/dpintaric/loan-prediction-elastic-net-logistic-regression)
> 5.  [Kaggle2](https://www.kaggle.com/code/stephenreed/lab-5-ridge-lasso-and-elasticnet-regressions)
> 6.  [Kaggle3](https://www.kaggle.com/code/kiyoung1027/linear-model-mlr-lasso-ridge-and-elastic-net/report)
> 7.  [Kaggle4](https://www.kaggle.com/code/jfeng1023/using-elastic-net-to-select-variables)
> 8.  [Kaggle5](https://www.kaggle.com/code/deepakkumargunjetti/introduction-to-elastic-net-regression)
> 9.  [Kaggle6](https://www.kaggle.com/code/uocoeeds/building-a-regression-model-with-elastic-net)
> 10.  [Book1](https://bradleyboehmke.github.io/HOML/regularized-regression.html)
> 11.  [Book2](https://scientistcafe.com/ids/r/ch10)
> 12.  [Book3](https://bookdown.org/ndirienzo/ista_321_data_mining/regularization.html)
> 13.  [Tuning Elastic Net](https://markvandewielblog.wordpress.com/2022/09/02/sparsity-can-not-be-estimated-why-tuning-the-elastic-net-is-hard/)

## code-1

```R
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
# Set multiple seeds
multiple_seeds <- function(seed) {
  set.seed(seed)
  list.of.fits <- list()
  for (i in 0:100) {
    fit.name <- paste0("alpha", i/100)
    list.of.fits[[fit.name]] <- cv.glmnet(
      x.train,
      y.train,
      alpha = i/100,
      standardize = TRUE,
      nfolds = 10,
      type.measure = "class",
      family = "binomial",
      parallel = TRUE
    )
  }
  
  results_df <- data.frame()
  
  for (fit.name in names(list.of.fits)) {
    fit <- list.of.fits[[fit.name]]
    lambda <- fit$lambda
    measure <- fit$cvm
    fit_df <- data.frame(seed = seed, alpha = fit.name, lambda = lambda, measure = measure)
    results_df <- rbind(results_df, fit_df)
  }
  
  return(results_df)
}

#total_seeds <- 10
results_list <- list()

for (s in 1:100) {
  results_list[[s]] <- multiple_seeds(s)
}
results_df <- do.call(rbind, results_list)
stopCluster(cl)
write.csv(results_df,"results_df.csv")
```

