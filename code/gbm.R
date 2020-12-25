library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)
library(RColorBrewer)       # Color selection for fancy tree plot
library(party)                  # Alternative decision tree algorithm
library(partykit)               # Convert rpart object to BinaryTree   
library(ROCit)
library(argparser)

main_dir <- '../model_results'
sub_dir <- 'gbm'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("gbm Dir already exists!")
}

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input", help="balanced data csv file",default = "../data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--training_rds", help="only training",default = "../model_results/gbm/gbm_train" )
p <- add_argument(p, "--training_and_val_rds", help="training and valuation",default = "../model_results/gbm/gbm_tv")
p <- add_argument(p, "--val_eval_table", help="only training",default = "../model_results/gbm/cnf_gbm__train.csv" )
p <- add_argument(p, "--testing_eval_table", help="training and valuation",default = "../model_results/gbm/cnf_gbm_tv.csv")
p <- add_argument(p, "--val_ROC", help="only training",default = "../model_results/gbm/gbm_train" )
p <- add_argument(p, "--testing_ROC", help="training and valuation",default = "../model_results/gbm/gbm_tv")

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

#gbm
library(gbm)
for(i in 1:10){
  num_gbm_train <- gbm(fraudulent ~ .,
                       data=trainData[[i]], 
                       distribution="bernoulli",
                       n.trees=200,interaction.depth=4,
                       shrinkage=0.05,verbose = T)
  num_gbm_tv <- gbm(fraudulent ~ .,
                    data=tvData[[i]], 
                    distribution="bernoulli",
                    n.trees=200,interaction.depth=4,
                    shrinkage=0.05,verbose = T)
  saveRDS(num_gbm_train,file=paste(args$training_rds,i,".rds",sep=""))
  saveRDS(num_gbm_tv,file=paste(args$training_and_val_rds,i,".rds",sep=""))
}
gbm_train_out <- list()
gbm_tv_out <- list()
pred_gbm_train <- list()
pred_gbm_tv <- list()
predbin_gbm_train <- list()
predbin_gbm_tv <- list()
factor_gbm_train <- list()
factor_gbm_tv <- list()
confusematrix_gbm_train <- list()
confusematrix_gbm_tv <- list()
acc.gbm_train <- list()
acc.gbm_tv <- list()
sens.gbm_train <- list()
sens.gbm_tv <- list()
spec.gbm_train <- list()
spec.gbm_tv <- list()
prec.gbm_train <- list()
prec.gbm_tv <- list()
rec.gbm_train <- list()
rec.gbm_tv <- list()
f1.gbm_train <- list()
f1.gbm_tv <- list()
balacc.gbm_train <- list()
balacc.gbm_tv <- list()
fold_gbm <- list()
cgbm_train <- list()
cgbm_tv <- list()
cgbm_train_final <- data.frame()
cgbm_tv_final <- data.frame()
roc.gbm_train <- list()
roc.gbm_tv <- list()

for(i in 1:10){
  num_gbm_train <- readRDS(file=paste(args$training_rds,i,".rds",sep=""))
  num_gbm_tv <- readRDS(file=paste(args$training_and_val_rds,i,".rds",sep=""))
  gbm_train_out[[i]] <-  predict(num_gbm_train, newdata=validData[[i]], type="response")
  gbm_tv_out[[i]] <-  predict(num_gbm_tv, newdata=testData[[i]], type="response")
  pred_gbm_train[[i]] <- ifelse(gbm_train_out[[i]]>0.5,1,0)
  pred_gbm_tv[[i]] <- ifelse(gbm_tv_out[[i]]>0.5,1,0)
  predbin_gbm_train[[i]] <- as.factor(pred_gbm_train[[i]])
  predbin_gbm_tv[[i]] <- as.factor(pred_gbm_tv[[i]])
  factor_gbm_train[[i]] <- as.factor(validData[[i]]$fraudulent)
  factor_gbm_tv[[i]] <- as.factor(testData[[i]]$fraudulent)
  confusematrix_gbm_train[[i]] <- confusionMatrix(predbin_gbm_train[[i]],factor_gbm_train[[i]])
  confusematrix_gbm_tv[[i]] <- confusionMatrix(predbin_gbm_tv[[i]],factor_gbm_tv[[i]])
  acc.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$overall[[1]]
  acc.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$overall[[1]]
  sens.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[1]]
  sens.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[1]]
  spec.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[2]]
  spec.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[2]]
  prec.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[5]]
  prec.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[5]]
  rec.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[6]]
  rec.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[6]]
  f1.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[7]]
  f1.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[7]]
  balacc.gbm_train[[i]] <- confusematrix_gbm_train[[i]]$byClass[[11]]
  balacc.gbm_tv[[i]] <- confusematrix_gbm_tv[[i]]$byClass[[11]]
  fold_gbm[[i]] <- paste0("fold",i)
  cgbm_train[[i]] <- data.frame(fold_gbm[[i]],round(acc.gbm_train[[i]],2),round(sens.gbm_train[[i]],2),round(spec.gbm_train[[i]],2),
                                round(prec.gbm_train[[i]],2),round(rec.gbm_train[[i]],2),round(f1.gbm_train[[i]],2),
                                round(balacc.gbm_train[[i]],2))
  names(cgbm_train[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")
  cgbm_train_final <- rbind(cgbm_train_final,cgbm_train[[i]])
  cgbm_tv[[i]] <- data.frame(fold_gbm[[i]],round(acc.gbm_train[[i]],2),round(sens.gbm_tv[[i]],2),round(spec.gbm_tv[[i]],2),
                             round(prec.gbm_tv[[i]],2),round(rec.gbm_tv[[i]],2),round(f1.gbm_tv[[i]],2),
                             round(balacc.gbm_tv[[i]],2))
  names(cgbm_tv[[i]]) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")
  cgbm_tv_final <- rbind(cgbm_tv_final,cgbm_tv[[i]])
  roc.gbm_train[[i]] <- rocit(pred_gbm_train[[i]], validData[[i]]$fraudulent)
  roc.gbm_tv[[i]] <- rocit(pred_gbm_tv[[i]], testData[[i]]$fraudulent)
}
write.csv(cgbm_train_final,file=args$val_eval_table)
write.csv(cgbm_tv_final,file=args$testing_eval_table)
for (i in 1:10){
  png(filename=paste(args$val_ROC,i,".png",sep=""))
  plot(roc.gbm_train[[i]], YIndex = F, values = F)
  dev.off()
}
for (i in 1:10){
  png(filename=paste(args$testing_ROC,i,".png",sep=""))
  plot(roc.gbm_tv[[i]], YIndex = F, values = F)
  dev.off()
}
