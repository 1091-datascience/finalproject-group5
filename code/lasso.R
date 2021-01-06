library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)
library(ROCit)
library(argparser)

main_dir <- './model_results'
sub_dir <- 'lasso'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("lasso Dir already exists!")
}

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input", help="balanced data csv file",default = "./data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--training_rds", help="only training",default = "./model_results/lasso/lasso_train" )
p <- add_argument(p, "--training_and_val_rds", help="training and valuation",default = "./model_results/lasso/lasso_tv")
p <- add_argument(p, "--training_cv_rds", help="only training",default = "./model_results/lasso/lasso_train_cv" )
p <- add_argument(p, "--training_and_val_cv_rds", help="training and valuation",default = "./model_results/lasso/lasso_tv_cv")
p <- add_argument(p, "--val_eval_table", help="only training",default = "./model_results/lasso/cnf_lasso__train.csv" )
p <- add_argument(p, "--testing_eval_table", help="training and valuation",default = "./model_results/lasso/cnf_lasso_tv.csv")
p <- add_argument(p, "--val_ROC", help="only training",default = "./model_results/lasso/lasso_train" )
p <- add_argument(p, "--testing_ROC", help="training and valuation",default = "./model_results/lasso/lasso_tv")

# trailingOnly å¦‚æ?œæ˜¯TRUE??„è©±ï¼Œæ?ƒåªç·¨è¼¯command-line?‡º?¾args??„å€¼args <- 
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

#lasso

library(glmnet)

for(i in 1:10){
  num_lasso_train=glmnet(x = data.matrix(trainData[[i]][, -length(trainData[[i]])]), 
                         y = trainData[[i]]$fraudulent, 
                         alpha = 1,
                         family = "binomial")
  cv_lasso_train = cv.glmnet(x = data.matrix(trainData[[i]][, -length(trainData[[i]])]), 
                             y = trainData[[i]]$fraudulent, 
                             alpha = 1,
                             family = "binomial")
  num_lasso_tv=glmnet(x = data.matrix(tvData[[i]][, -length(tvData[[i]])]), 
                      y = tvData[[i]]$fraudulent, 
                      alpha = 1,
                      family = "binomial")
  cv_lasso_tv = cv.glmnet(x = data.matrix(tvData[[i]][, -length(tvData[[i]])]), 
                          y = tvData[[i]]$fraudulent, 
                          alpha = 1,
                          family = "binomial")
  saveRDS(num_lasso_train,file=paste(args$training_rds,i,".rds",sep=""))
  saveRDS(cv_lasso_train,file=paste(args$training_cv_rds,i,".rds",sep=""))
  saveRDS(num_lasso_tv,file=paste(args$training_and_val_rds,i,".rds",sep=""))
  saveRDS(cv_lasso_tv,file=paste(args$training_and_val_cv_rds,i,".rds",sep=""))
}
lasso_train_out <- list()
lasso_tv_out <- list()
pred_lasso_train <- list()
pred_lasso_tv <- list()
predbin_lasso_train <- list()
predbin_lasso_tv <- list()
factor_lasso_train <- list()
factor_lasso_tv <- list()
confusematrix_lasso_train <- list()
confusematrix_lasso_tv <- list()
acc.lasso_train <- list()
acc.lasso_tv <- list()
sens.lasso_train <- list()
sens.lasso_tv <- list()
spec.lasso_train <- list()
spec.lasso_tv <- list()
prec.lasso_train <- list()
prec.lasso_tv <- list()
rec.lasso_train <- list()
rec.lasso_tv <- list()
f1.lasso_train <- list()
f1.lasso_tv <- list()
balacc.lasso_train <- list()
balacc.lasso_tv <- list()
fold_lasso <- list()
classo_train <- list()
classo_tv <- list()
classo_train_final <- data.frame()
classo_tv_final <- data.frame()
roc.lasso_train <- list()
roc.lasso_tv <- list()
auc.lasso_train <- list()
auc.lasso_tv <- list()
for(i in 1:10){
  num_lasso_train <- readRDS(file=paste(args$training_rds,i,".rds",sep=""))
  num_lasso_tv <- readRDS(file=paste(args$training_and_val_rds,i,".rds",sep=""))
  cv_lasso_train <- readRDS(file=paste(args$training_cv_rds,i,".rds",sep=""))
  cv_lasso_tv <- readRDS(file=paste(args$training_and_val_cv_rds,i,".rds",sep=""))
  best_train.lasso.lambda =cv_lasso_train$lambda.min
  best_tv.lasso.lambda =cv_lasso_train$lambda.min
  lasso_train_out[[i]] <-  predict(num_lasso_train, s = best_train.lasso.lambda, 
                                   newx = data.matrix(validData[[i]][, -length(validData[[i]])]))
  lasso_tv_out[[i]] <-  predict(num_lasso_tv, s = best_tv.lasso.lambda, 
                                newx = data.matrix(testData[[i]][, -length(testData[[i]])]))
  pred_lasso_train[[i]] <- ifelse(lasso_train_out[[i]]>0,1,0)
  pred_lasso_tv[[i]] <- ifelse(lasso_tv_out[[i]]>0,1,0)
  predbin_lasso_train[[i]] <- as.factor(pred_lasso_train[[i]])
  predbin_lasso_tv[[i]] <- as.factor(pred_lasso_tv[[i]])
  factor_lasso_train[[i]] <- as.factor(validData[[i]]$fraudulent)
  factor_lasso_tv[[i]] <- as.factor(testData[[i]]$fraudulent)
  confusematrix_lasso_train[[i]] <- confusionMatrix(predbin_lasso_train[[i]],factor_lasso_train[[i]])
  confusematrix_lasso_tv[[i]] <- confusionMatrix(predbin_lasso_tv[[i]],factor_lasso_tv[[i]])
  acc.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$overall[[1]]
  acc.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$overall[[1]]
  sens.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[1]]
  sens.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[1]]
  spec.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[2]]
  spec.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[2]]
  prec.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[5]]
  prec.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[5]]
  rec.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[6]]
  rec.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[6]]
  f1.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[7]]
  f1.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[7]]
  balacc.lasso_train[[i]] <- confusematrix_lasso_train[[i]]$byClass[[11]]
  balacc.lasso_tv[[i]] <- confusematrix_lasso_tv[[i]]$byClass[[11]]
  roc.lasso_train[[i]] <- rocit(as.numeric(pred_lasso_train[[i]]), validData[[i]]$fraudulent)
  roc.lasso_tv[[i]] <- rocit(as.numeric(pred_lasso_tv[[i]]), testData[[i]]$fraudulent)
  auc.lasso_train[[i]] <- as.numeric(ciAUC(roc.lasso_train[[i]])[1])
  auc.lasso_tv[[i]] <- as.numeric(ciAUC(roc.lasso_tv[[i]])[1])
  fold_lasso[[i]] <- paste0("fold",i)
  classo_train[[i]] <- data.frame(fold_lasso[[i]],round(acc.lasso_train[[i]],4),round(sens.lasso_train[[i]],4),round(spec.lasso_train[[i]],4),
                                round(prec.lasso_train[[i]],4),round(rec.lasso_train[[i]],4),round(f1.lasso_train[[i]],4),
                                round(balacc.lasso_train[[i]],4),round(auc.lasso_train[[i]],4))
  names(classo_train[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  classo_train_final <- rbind(classo_train_final,classo_train[[i]])
  classo_tv[[i]] <- data.frame(fold_lasso[[i]],round(acc.lasso_train[[i]],4),round(sens.lasso_tv[[i]],4),round(spec.lasso_tv[[i]],4),
                             round(prec.lasso_tv[[i]],4),round(rec.lasso_tv[[i]],4),round(f1.lasso_tv[[i]],4),
                             round(balacc.lasso_tv[[i]],4),round(auc.lasso_tv[[i]],4))
  names(classo_tv[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  classo_tv_final <- rbind(classo_tv_final,classo_tv[[i]])
  
} 
l <- list()
h <- list()
for(s in 2:length(classo_train_final)){
  l[[s]] <- mean(classo_train_final[[s]])
}
k <- data.frame(t(c("ave.",round(l[[2]],2),
                    round(l[[3]],2),round(l[[4]],2),round(l[[5]],2)
                    ,round(l[[6]],2),round(l[[7]],2),round(l[[8]],2)
                    ,round(l[[9]],2))))
names(k) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
classo_train_final <- rbind(classo_train_final,k)
for(s in 2:length(classo_tv_final)){
  h[[s]] <- mean(classo_tv_final[[s]])
}
q <- data.frame(t(c("ave.",round(h[[2]],2),
                    round(h[[3]],2),round(h[[4]],2),round(h[[5]],2)
                    ,round(h[[6]],2),round(h[[7]],2),round(h[[8]],2)
                    ,round(h[[9]],2))))
names(q) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
classo_tv_final <- rbind(classo_tv_final,k)

write.csv(classo_train_final,file=args$val_eval_table)
write.csv(classo_tv_final,file=args$testing_eval_table)
for (i in 1:10){
  png(filename=paste(args$val_ROC,i,".png",sep=""))
  plot(roc.lasso_train[[i]], YIndex = F, values = F)
  dev.off()
}
for (i in 1:10){
  png(filename=paste(args$testing_ROC,i,".png",sep=""))
  plot(roc.lasso_tv[[i]], YIndex = F, values = F)
  dev.off()
}
