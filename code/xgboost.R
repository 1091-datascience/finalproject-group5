library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)  
library(ROCit)
library(argparser)

main_dir <- './model_results'
sub_dir <- 'xgb'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("xgb Dir already exists!")
}

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input", help="balanced data csv file",default = "./data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--training_rds", help="only training",default = "./model_results/xgb/xgb_train" )
p <- add_argument(p, "--training_and_val_rds", help="training and valuation",default = ".//model_results/xgb/xgb_tv")
p <- add_argument(p, "--val_eval_table", help="only training",default = "./model_results/xgb/cnf_xgb_train.csv" )
p <- add_argument(p, "--testing_eval_table", help="training and valuation",default = "./model_results/xgb/cnf_xgb_tv.csv")
p <- add_argument(p, "--val_ROC", help="only training",default = "./model_results/xgb/xgb_train" )
p <- add_argument(p, "--testing_ROC", help="training and valuation",default = "./model_results/xgb/xgb_tv")

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

#xGboost
library(xgboost)
dtrain <- list()
dval <- list()
dtv <- list()
dtest <- list()
for(i in 1:10){
  dtrain[[i]] = xgb.DMatrix(data = as.matrix((trainData[[i]][, -length(trainData[[i]])])),
                            label =trainData[[i]]$fraudulent)
  dval[[i]] = xgb.DMatrix(data = as.matrix((validData[[i]][, -length(validData[[i]])])),
                          label =validData[[i]]$fraudulent)
  dtv[[i]] = xgb.DMatrix(data = as.matrix((tvData[[i]][, -length(tvData[[i]])])),
                         label =tvData[[i]]$fraudulent)
  dtest[[i]] = xgb.DMatrix(data = as.matrix((testData[[i]][, -length(testData[[i]])])),
                           label =testData[[i]]$fraudulent)
}
for(i in 1:10){
  num_xgb_train <- xgb.train(data=dtrain[[i]],max.depth=4,eta=.2,nthread=6,nround=50,objective="binary:logistic",
                             eval_metric="error",eval_metric="logloss",eval_metric="auc")
  num_xgb_tv <- xgb.train(data=dtv[[i]],max.depth=4,eta=.2,nthread=6,nround=50,objective="binary:logistic",
                          eval_metric="error",eval_metric="logloss",eval_metric="auc")
  saveRDS(num_xgb_train,file=paste(args$training_rds,i,".rds",sep=""))
  saveRDS(num_xgb_tv,file=paste(args$training_and_val_rds,i,".rds",sep=""))
}
xgb_train_out <- list()
xgb_tv_out <- list()
pred_xgb_train <- list()
pred_xgb_tv <- list()
predbin_xgb_train <- list()
predbin_xgb_tv <- list()
factor_xgb_train <- list()
factor_xgb_tv <- list()
confusematrix_xgb_train <- list()
confusematrix_xgb_tv <- list()
acc.xgb_train <- list()
acc.xgb_tv <- list()
sens.xgb_train <- list()
sens.xgb_tv <- list()
spec.xgb_train <- list()
spec.xgb_tv <- list()
prec.xgb_train <- list()
prec.xgb_tv <- list()
rec.xgb_train <- list()
rec.xgb_tv <- list()
f1.xgb_train <- list()
f1.xgb_tv <- list()
balacc.xgb_train <- list()
balacc.xgb_tv <- list()
fold_xgb <- list()
cxgb_train <- list()
cxgb_tv <- list()
cxgb_train_final <- data.frame()
cxgb_tv_final <- data.frame()
roc.xgb_train <- list()
roc.xgb_tv <- list()
auc.xgb_train <- list()
auc.xgb_tv <- list()
for(i in 1:10){
  num_xgb_train <- readRDS(file=paste(args$training_rds,i,".rds",sep=""))
  num_xgb_tv <- readRDS(file=paste(args$training_and_val_rds,i,".rds",sep=""))
  xgb_train_out[[i]] <-  predict(num_xgb_train, newdata=dval[[i]])
  xgb_tv_out[[i]] <-  predict(num_xgb_tv, newdata=dtest[[i]])
  pred_xgb_train[[i]] <- ifelse(xgb_train_out[[i]]>0.5,1,0)
  pred_xgb_tv[[i]] <- ifelse(xgb_tv_out[[i]]>0.5,1,0)
  predbin_xgb_train[[i]] <- as.factor(pred_xgb_train[[i]])
  predbin_xgb_tv[[i]] <- as.factor(pred_xgb_tv[[i]])
  factor_xgb_train[[i]] <- as.factor(validData[[i]]$fraudulent)
  factor_xgb_tv[[i]] <- as.factor(testData[[i]]$fraudulent)
  confusematrix_xgb_train[[i]] <- confusionMatrix(predbin_xgb_train[[i]],factor_xgb_train[[i]])
  confusematrix_xgb_tv[[i]] <- confusionMatrix(predbin_xgb_tv[[i]],factor_xgb_tv[[i]])
  acc.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$overall[[1]]
  acc.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$overall[[1]]
  sens.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[1]]
  sens.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[1]]
  spec.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[2]]
  spec.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[2]]
  prec.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[5]]
  prec.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[5]]
  rec.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[6]]
  rec.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[6]]
  f1.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[7]]
  f1.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[7]]
  balacc.xgb_train[[i]] <- confusematrix_xgb_train[[i]]$byClass[[11]]
  balacc.xgb_tv[[i]] <- confusematrix_xgb_tv[[i]]$byClass[[11]]
  roc.xgb_train[[i]] <- rocit(pred_xgb_train[[i]], validData[[i]]$fraudulent)
  roc.xgb_tv[[i]] <- rocit(pred_xgb_tv[[i]], testData[[i]]$fraudulent)
  auc.xgb_train[[i]] <- as.numeric(ciAUC(roc.xgb_train[[i]])[1])
  auc.xgb_tv[[i]] <- as.numeric(ciAUC(roc.xgb_tv[[i]])[1])
  fold_xgb[[i]] <- paste0("fold",i)
  cxgb_train[[i]] <- data.frame(fold_xgb[[i]],round(acc.xgb_train[[i]],4),round(sens.xgb_train[[i]],4),round(spec.xgb_train[[i]],4),
                                  round(prec.xgb_train[[i]],4),round(rec.xgb_train[[i]],4),round(f1.xgb_train[[i]],4),
                                  round(balacc.xgb_train[[i]],4),round(auc.xgb_train[[i]],4))
  names(cxgb_train[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  cxgb_train_final <- rbind(cxgb_train_final,cxgb_train[[i]])
  cxgb_tv[[i]] <- data.frame(fold_xgb[[i]],round(acc.xgb_train[[i]],4),round(sens.xgb_tv[[i]],4),round(spec.xgb_tv[[i]],4),
                               round(prec.xgb_tv[[i]],4),round(rec.xgb_tv[[i]],4),round(f1.xgb_tv[[i]],4),
                               round(balacc.xgb_tv[[i]],4),round(auc.xgb_tv[[i]],4))
  names(cxgb_tv[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  cxgb_tv_final <- rbind(cxgb_tv_final,cxgb_tv[[i]])
  
} 
l <- list()
h <- list()
for(s in 2:length(cxgb_train_final)){
  l[[s]] <- mean(cxgb_train_final[[s]])
}
k <- data.frame(t(c("ave.",round(l[[2]],2),
                    round(l[[3]],2),round(l[[4]],2),round(l[[5]],2)
                    ,round(l[[6]],2),round(l[[7]],2),round(l[[8]],2)
                    ,round(l[[9]],2))))
names(k) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
cxgb_train_final <- rbind(cxgb_train_final,k)
for(s in 2:length(cxgb_tv_final)){
  h[[s]] <- mean(cxgb_tv_final[[s]])
}
q <- data.frame(t(c("ave.",round(h[[2]],2),
                    round(h[[3]],2),round(h[[4]],2),round(h[[5]],2)
                    ,round(h[[6]],2),round(h[[7]],2),round(h[[8]],2)
                    ,round(h[[9]],2))))
names(q) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
cxgb_tv_final <- rbind(cxgb_tv_final,k)

write.csv(cxgb_train_final,file=args$val_eval_table)
write.csv(cxgb_tv_final,file=args$testing_eval_table)
for (i in 1:10){
  png(filename=paste(args$val_ROC,i,".png",sep=""))
  plot(roc.xgb_train[[i]], YIndex = F, values = F)
  dev.off()
}
for (i in 1:10){
  png(filename=paste(args$testing_ROC,i,".png",sep=""))
  plot(roc.xgb_tv[[i]], YIndex = F, values = F)
  dev.off()
}
