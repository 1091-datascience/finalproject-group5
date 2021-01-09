library(psych)  #for general functions
library(ggplot2)  #for data visualization
library(caret)#for training and cross validation (also calls other model libaries)
library(ROCit)
library(argparser)
library(gbm)

main_dir <- './model_results'
sub_dir <- 'unb'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("unb Dir already exists!")
}

p <- arg_parser("Unbalanced data training model use in balanced testing data")
p <- add_argument(p, "--input", help="unbalanced data csv file",default = "./data/fake_job_postings_TFIDF.csv")
p <- add_argument(p, "--training_rds", help="training model weight",default = "./model_results/unb/gbm_ub.rds")
p <- add_argument(p, "--output", help="evaluation table",default = "./model_results/unb/cnf_gbm_unb.csv")
#  trailingOnly 如果是TRUE的話，會只編輯command-line出現args的值args <- 
args <- parse_args(p, commandArgs(trailingOnly = TRUE))
df1<-read.csv(args$input_csv,fileEncoding='utf-8')
df2 <- read.csv("./data/fake_job_postings_TFIDF_balance.csv")
set.seed(20201219)
index <-  sort(sample(nrow(df1), nrow(df1)*.75))
train <- df1[index,][,-1]
test <-  df1[-index,][,-1]
train.gbm_ub <- gbm(train$fraudulent ~ .,
                    data=train, distribution="bernoulli",
                    n.trees=1000,interaction.depth=4,
                    shrinkage=0.05,verbose = T)
saveRDS(train.gbm_ub,args$model_weight)

my_model_gbm <- readRDS(args$model_weight)
pred_gbm <- predict(object=my_model_gbm,newdata=test,type="response")
pred_gbm_b <- predict(object=my_model_gbm,newdata=df2,type="response")
predbin_gbm <- as.factor(ifelse(pred_gbm>0.5,1,0))
predbin_gbm_b <- as.factor(ifelse(pred_gbm_b>0.5,1,0))
k <- as.factor(test$fraudulent)
h <-  as.factor(df2$fraudulent)
confusematrix_gbm <- confusionMatrix(predbin_gbm,k)
confusematrix_gbm_b <- confusionMatrix(predbin_gbm_b,h)
acc.confusematrix_gbm <- confusematrix_gbm$overall[[1]]
sens.confusematrix_gbm <- confusematrix_gbm$byClass[[1]]
spec.confusematrix_gbm <- confusematrix_gbm$byClass[[2]]
prec.confusematrix_gbm <- confusematrix_gbm$byClass[[5]]
rec.confusematrix_gbm <- confusematrix_gbm$byClass[[6]]
f1.confusematrix_gbm <- confusematrix_gbm$byClass[[7]]
balacc.confusematrix_gbm <- confusematrix_gbm$byClass[[11]]
roc.gbm <- rocit(pred_gbm, test$fraudulent)
auc.gbm <- as.numeric(ciAUC(roc.gbm)[1])
acc.confusematrix_gbm_b <- confusematrix_gbm_b$overall[[1]]
sens.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[1]]
spec.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[2]]
prec.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[5]]
rec.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[6]]
f1.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[7]]
balacc.confusematrix_gbm_b <- confusematrix_gbm_b$byClass[[11]]
roc.gbm_b <- rocit(pred_gbm_b, test$fraudulent)
auc.gbm_b <- as.numeric(ciAUC(roc.gbm_b)[1])
cgbm <- data.frame("ub_model_test_ub",round(acc.confusematrix_gbm,2),round(sens.confusematrix_gbm,2),round(spec.confusematrix_gbm,2),
                   round(prec.confusematrix_gbm,2),round(rec.confusematrix_gbm,2),round(f1.confusematrix_gbm,2),
                   round(balacc.confusematrix_gbm,2),round(auc.gbm,2))
cgbm_b <- data.frame(t("ub_model_test_b",round(acc.confusematrix_gbm_b,2),round(sens.confusematrix_gbm_b,2),round(spec.confusematrix_gbm_b,2),
                   round(prec.confusematrix_gbm_b,2),round(rec.confusematrix_gbm_b,2),round(f1.confusematrix_gbm_b,2),
                   round(balacc.confusematrix_gbm_b,2),round(auc.gbm_b,2)))
names(cgbm) <- c("performance","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
names(cgbm_b) <- c("performance","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy","auc")
cgbm_final <- rbind(cgbm,cgbm_b)
write.csv(cgbm_final,file=args$output_csv)
png(filename=paste("gbm_unb",".png",sep=""))
plot(roc.gbm, YIndex = F, values = F)
dev.off()
