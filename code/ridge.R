library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)
library(RColorBrewer)       # Color selection for fancy tree plot
library(party)                  # Alternative decision tree algorithm
library(partykit)               # Convert rpart object to BinaryTree   
library(ROCit)
library(argparser)

main_dir <- '../model_results'
sub_dir <- 'ridge'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("ridge Dir already exists!")
}

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input", help="balanced data csv file",default = "../data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--training_rds", help="only training",default = "../model_results/ridge/ridge_train" )
p <- add_argument(p, "--training_and_val_rds", help="training and valuation",default = "../model_results/ridge/ridge_train_cv")
p <- add_argument(p, "--training_cv_rds", help="only training",default = "../model_results/ridge/ridge_tv" )
p <- add_argument(p, "--training_and_val_cv_rds", help="training and valuation",default = "../model_results/ridge/ridge_tv_cv")
p <- add_argument(p, "--val_eval_table", help="only training",default = "../model_results/ridge/cnf_ridge_train.csv" )
p <- add_argument(p, "--testing_eval_table", help="training and valuation",default = "../model_results/ridge/cnf_ridge_tv.csv")
p <- add_argument(p, "--val_ROC", help="only training",default = "../model_results/ridge/ridge_train" )
p <- add_argument(p, "--testing_ROC", help="training and valuation",default = "../model_results/ridge/ridge_tv")

# trailingOnly 如果是TRUE的話，會只編輯command-line出現args的值args <- 
args <- parse_args(p, commandArgs(trailingOnly = TRUE))

df2 <- read.csv(args$input)
table(df2$fraudulent)
df2 <- df2[,-1]
df2 <- df2[sample(nrow(df2)),]
folds <- cut(seq(1,nrow(df2)),breaks=as.numeric(10),labels=FALSE)
testIndexes <- list()
validIndexes <- list()
testData <- list()
validData <- list()
trainData <- list()
tvData <- list()

for(i in 1:10){
  if(i==10){
    testIndexes[[i]] <- which(folds==i,arr.ind=TRUE)
    validIndexes[[i]] <- which(folds==1,arr.ind=TRUE)
    testData[[i]] <- df2[testIndexes[[i]], ]
    validData[[i]] <- df2[validIndexes[[i]], ]
    trainData[[i]] <- df2[-rbind(testIndexes[[i]],validIndexes[[i]]),]
  }else{
    testIndexes[[i]] <- which(folds==i,arr.ind=TRUE)
    validIndexes[[i]] <- which(folds==i+1,arr.ind=TRUE)
    testData[[i]] <- df2[testIndexes[[i]], ]
    validData[[i]] <- df2[validIndexes[[i]], ]
    trainData[[i]] <- df2[-rbind(testIndexes[[i]],validIndexes[[i]]),]
  }
}
for(i in 1:10){
  tvData[[i]] <- rbind.data.frame(trainData[[i]],validData[[i]])
}

#ridge

for(i in 1:10){
  num_ridge_train=glmnet(x = data.matrix(trainData[[i]][, -length(trainData[[i]])]), 
                         y = trainData[[i]]$fraudulent, 
                         alpha = 0,
                         family = "binomial")
  cv_ridge_train = cv.glmnet(x = data.matrix(trainData[[i]][, -length(trainData[[i]])]), 
                             y = trainData[[i]]$fraudulent, 
                             alpha = 0,
                             family = "binomial")
  num_ridge_tv=glmnet(x = data.matrix(tvData[[i]][, -length(tvData[[i]])]), 
                      y = tvData[[i]]$fraudulent, 
                      alpha = 0,
                      family = "binomial")
  cv_ridge_tv = cv.glmnet(x = data.matrix(tvData[[i]][, -length(tvData[[i]])]), 
                          y = tvData[[i]]$fraudulent, 
                          alpha = 0,
                          family = "binomial")
  saveRDS(num_ridge_train,file=paste(args$training_rds,i,".rds",sep=""))
  saveRDS(cv_ridge_train,file=paste(args$training_cv_rds,i,".rds",sep=""))
  saveRDS(num_ridge_tv,file=paste(args$training_and_val_rds,i,".rds",sep=""))
  saveRDS(cv_ridge_tv,file=paste(args$training_and_val_cv_rds,i,".rds",sep=""))
}
ridge_train_out <- list()
ridge_tv_out <- list()
pred_ridge_train <- list()
pred_ridge_tv <- list()
predbin_ridge_train <- list()
predbin_ridge_tv <- list()
factor_ridge_train <- list()
factor_ridge_tv <- list()
confusematrix_ridge_train <- list()
confusematrix_ridge_tv <- list()
acc.ridge_train <- list()
acc.ridge_tv <- list()
sens.ridge_train <- list()
sens.ridge_tv <- list()
spec.ridge_train <- list()
spec.ridge_tv <- list()
prec.ridge_train <- list()
prec.ridge_tv <- list()
rec.ridge_train <- list()
rec.ridge_tv <- list()
f1.ridge_train <- list()
f1.ridge_tv <- list()
balacc.ridge_train <- list()
balacc.ridge_tv <- list()
fold_ridge <- list()
cridge_train <- list()
cridge_tv <- list()
cridge_train_final <- data.frame()
cridge_tv_final <- data.frame()
roc.ridge_train <- list()
roc.ridge_tv <- list()
for(i in 1:10){
  num_ridge_train <- readRDS(file=paste(args$training_rds,i,".rds",sep=""))
  num_ridge_tv <- readRDS(file=paste(args$training_and_val_rds,i,".rds",sep=""))
  cv_ridge_train <- readRDS(file=paste(args$training_cv_rds,i,".rds",sep=""))
  cv_ridge_tv <- readRDS(file=paste(args$training_and_val_cv_rds,i,".rds",sep=""))
  best_train.ridge.lambda =cv_ridge_train$lambda.min
  best_tv.ridge.lambda =cv_ridge_train$lambda.min
  ridge_train_out[[i]] <-  predict(num_ridge_train, s = best_train.ridge.lambda, 
                                   newx = data.matrix(validData[[i]][, -length(validData[[i]])]))
  ridge_tv_out[[i]] <-  predict(num_ridge_tv, s = best_tv.ridge.lambda, 
                                newx = data.matrix(testData[[i]][, -length(testData[[i]])]))
  pred_ridge_train[[i]] <- ifelse(ridge_train_out[[i]]>0,1,0)
  pred_ridge_tv[[i]] <- ifelse(ridge_tv_out[[i]]>0,1,0)
  predbin_ridge_train[[i]] <- as.factor(pred_ridge_train[[i]])
  predbin_ridge_tv[[i]] <- as.factor(pred_ridge_tv[[i]])
  factor_ridge_train[[i]] <- as.factor(validData[[i]]$fraudulent)
  factor_ridge_tv[[i]] <- as.factor(testData[[i]]$fraudulent)
  confusematrix_ridge_train[[i]] <- confusionMatrix(predbin_ridge_train[[i]],factor_ridge_train[[i]])
  confusematrix_ridge_tv[[i]] <- confusionMatrix(predbin_ridge_tv[[i]],factor_ridge_tv[[i]])
  acc.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$overall[[1]]
  acc.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$overall[[1]]
  sens.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[1]]
  sens.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[1]]
  spec.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[2]]
  spec.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[2]]
  prec.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[5]]
  prec.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[5]]
  rec.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[6]]
  rec.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[6]]
  f1.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[7]]
  f1.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[7]]
  balacc.ridge_train[[i]] <- confusematrix_ridge_train[[i]]$byClass[[11]]
  balacc.ridge_tv[[i]] <- confusematrix_ridge_tv[[i]]$byClass[[11]]
  fold_ridge[[i]] <- paste0("fold",i)
  cridge_train[[i]] <- data.frame(fold_ridge[[i]],round(acc.ridge_train[[i]],2),round(sens.ridge_train[[i]],2),round(spec.ridge_train[[i]],2),
                                  round(prec.ridge_train[[i]],2),round(rec.ridge_train[[i]],2),round(f1.ridge_train[[i]],2),
                                  round(balacc.ridge_train[[i]],2))
  names(cridge_train[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")
  cridge_train_final <- rbind(cridge_train_final,cridge_train[[i]])
  cridge_tv[[i]] <- data.frame(fold_ridge[[i]],round(acc.ridge_train[[i]],2),round(sens.ridge_tv[[i]],2),round(spec.ridge_tv[[i]],2),
                               round(prec.ridge_tv[[i]],2),round(rec.ridge_tv[[i]],2),round(f1.ridge_tv[[i]],2),
                               round(balacc.ridge_tv[[i]],2))
  names(cridge_tv[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")
  cridge_tv_final <- rbind(cridge_tv_final,cridge_tv[[i]])
  roc.ridge_train[[i]] <- rocit(as.numeric(pred_ridge_train[[i]]), validData[[i]]$fraudulent)
  roc.ridge_tv[[i]] <- rocit(as.numeric(pred_ridge_tv[[i]]), testData[[i]]$fraudulent)
}
write.csv(cridge_train_final,file=args$val_eval_table)
write.csv(cridge_tv_final,file=args$testing_eval_table)
for (i in 1:10){
  png(filename=paste(args$val_ROC,i,".png",sep=""))
  plot(roc.ridge_train[[i]], YIndex = F, values = F)
  dev.off()
}
for (i in 1:10){
  png(filename=paste(args$testing_ROC,i,".png",sep=""))
  plot(roc.ridge_tv[[i]], YIndex = F, values = F)
  dev.off()
}