library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)
library(ROCit)
library(argparser)

main_dir <- './model_results'
sub_dir <- 'decision_tree'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("Decision tree Dir already exists!")
}
p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input", help="balanced data csv file",default = "./data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--training_rds", help="only training",default = "./model_results/decision_tree/dtree_train" )
p <- add_argument(p, "--training_and_val_rds", help="training and valuation",default = "./model_results/decision_tree/dtree_tv")
p <- add_argument(p, "--val_eval_table", help="only training",default = "./model_results/decision_tree/cnf_dtree_train.csv" )
p <- add_argument(p, "--testing_eval_table", help="training and valuation",default = "./model_results/decision_tree/cnf_dtree_tv.csv")
p <- add_argument(p, "--val_ROC", help="only training",default = "./model_results/decision_tree/dtree_train" )
p <- add_argument(p, "--testing_ROC", help="training and valuation",default = "./model_results/decision_tree/dtree_tv")

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


#decision tree
library(rpart)
#train
for(i in 1:10){
  num_tree_train = rpart(fraudulent ~ ., 
                         data = trainData[[i]], method="class", minbucket=5,
                         parms = list(split="information"))
  num_tree_tv = rpart(fraudulent ~ ., 
                      data = tvData[[i]], method="class", minbucket=5,
                      parms = list(split="information"))
  saveRDS(num_tree_train,file=paste(args$training_rds,i,".rds",sep=""))
  saveRDS(num_tree_tv,file=paste(args$training_and_val_rds,i,".rds",sep=""))
}
tree_train_out <- list()
tree_tv_out <- list()
pred_tree_train <- list()
pred_tree_tv <- list()
predbin_tree_train <- list()
predbin_tree_tv <- list()
factor_tree_train <- list()
factor_tree_tv <- list()
confusematrix_tree_train <- list()
confusematrix_tree_tv <- list()
acc.tree_train <- list()
acc.tree_tv <- list()
sens.tree_train <- list()
sens.tree_tv <- list()
spec.tree_train <- list()
spec.tree_tv <- list()
prec.tree_train <- list()
prec.tree_tv <- list()
rec.tree_train <- list()
rec.tree_tv <- list()
f1.tree_train <- list()
f1.tree_tv <- list()
balacc.tree_train <- list()
balacc.tree_tv <- list()
fold_tree <- list()
ctree_train <- list()
ctree_tv <- list()
ctree_train_final <- data.frame()
ctree_tv_final <- data.frame()
roc.tree_train <- list()
roc.tree_tv <- list()
plt_roc_tree_train <- list()
plt_roc_tree_tv <- list()
auc.tree_train <- list()
auc.tree_tv <- list()
for(i in 1:10){
  num_tree_train <- readRDS(file=paste(args$training_rds,i,".rds",sep=""))
  num_tree_tv <- readRDS(file=paste(args$training_and_val_rds,i,".rds",sep=""))
  tree_train_out[[i]] <-  predict(num_tree_train, newdata=validData[[i]], type="prob")
  tree_tv_out[[i]] <-  predict(num_tree_tv, newdata=testData[[i]], type="prob")
  pred_tree_train[[i]] <- ifelse(tree_train_out[[i]]>0.5,0,1)
  pred_tree_tv[[i]] <- ifelse(tree_tv_out[[i]]>0.5,0,1)
  predbin_tree_train[[i]] <- as.factor(pred_tree_train[[i]][,1])
  predbin_tree_tv[[i]] <- as.factor(pred_tree_tv[[i]][,1])
  factor_tree_train[[i]] <- as.factor(validData[[i]]$fraudulent)
  factor_tree_tv[[i]] <- as.factor(testData[[i]]$fraudulent)
  confusematrix_tree_train[[i]] <- confusionMatrix(predbin_tree_train[[i]],factor_tree_train[[i]])
  confusematrix_tree_tv[[i]] <- confusionMatrix(predbin_tree_tv[[i]],factor_tree_tv[[i]])
  acc.tree_train[[i]] <- confusematrix_tree_train[[i]]$overall[[1]]
  acc.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$overall[[1]]
  sens.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[1]]
  sens.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[1]]
  spec.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[2]]
  spec.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[2]]
  prec.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[5]]
  prec.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[5]]
  rec.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[6]]
  rec.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[6]]
  f1.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[7]]
  f1.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[7]]
  balacc.tree_train[[i]] <- confusematrix_tree_train[[i]]$byClass[[11]]
  balacc.tree_tv[[i]] <- confusematrix_tree_tv[[i]]$byClass[[11]]
  roc.tree_train[[i]] <- rocit(pred_tree_train[[i]][,1], validData[[i]]$fraudulent)
  roc.tree_tv[[i]] <- rocit(pred_tree_tv[[i]][,1], testData[[i]]$fraudulent)
  auc.tree_train[[i]] <- as.numeric(ciAUC(roc.tree_train[[i]])[1])
  auc.tree_tv[[i]] <- as.numeric(ciAUC(roc.tree_tv[[i]])[1])
  fold_tree[[i]] <- paste0("fold",i)
  ctree_train[[i]] <- data.frame(fold_tree[[i]],round(acc.tree_train[[i]],4),round(sens.tree_train[[i]],4),round(spec.tree_train[[i]],4),
                                 round(prec.tree_train[[i]],4),round(rec.tree_train[[i]],4),round(f1.tree_train[[i]],4),
                                 round(balacc.tree_train[[i]],4),round(auc.tree_train[[i]],4))
  names(ctree_train[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  ctree_train_final <- rbind(ctree_train_final,ctree_train[[i]])
  ctree_tv[[i]] <- data.frame(fold_tree[[i]],round(acc.tree_train[[i]],4),round(sens.tree_tv[[i]],4),round(spec.tree_tv[[i]],4),
                              round(prec.tree_tv[[i]],4),round(rec.tree_tv[[i]],4),round(f1.tree_tv[[i]],4),
                              round(balacc.tree_tv[[i]],4),round(auc.tree_tv[[i]],4))
  names(ctree_tv[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
  ctree_tv_final <- rbind(ctree_tv_final,ctree_tv[[i]])
  
} 
l <- list()
h <- list()
for(s in 2:length(ctree_train_final)){
  l[[s]] <- mean(ctree_train_final[[s]])
}
k <- data.frame(t(c("ave.",round(l[[2]],2),
                    round(l[[3]],2),round(l[[4]],2),round(l[[5]],2)
                    ,round(l[[6]],2),round(l[[7]],2),round(l[[8]],2)
                    ,round(l[[9]],2))))
names(k) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
ctree_train_final <- rbind(ctree_train_final,k)
for(s in 2:length(ctree_tv_final)){
  h[[s]] <- mean(ctree_tv_final[[s]])
}
q <- data.frame(t(c("ave.",round(h[[2]],2),
                    round(h[[3]],2),round(h[[4]],2),round(h[[5]],2)
                    ,round(h[[6]],2),round(h[[7]],2),round(h[[8]],2)
                    ,round(h[[9]],2))))
names(q) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
ctree_tv_final <- rbind(ctree_tv_final,k)
write.csv(ctree_train_final,file=args$val_eval_table)
write.csv(ctree_tv_final,file=args$testing_eval_table)
for (i in 1:10){
  png(filename=paste(args$val_ROC,i,".png",sep=""))
  plot(roc.tree_train[[i]], YIndex = F, values = F)
  dev.off()
}
for (i in 1:10){
  png(filename=paste(args$testing_ROC,i,".png",sep=""))
  plot(roc.tree_tv[[i]], YIndex = F, values = F)
  dev.off()
}

