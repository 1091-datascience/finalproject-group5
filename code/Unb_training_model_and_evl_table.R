library(gbm) 
library(argparser)


main_dir <- '../model_results'
sub_dir <- 'unb'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("unb Dir already exists!")
}

p <- arg_parser("Unbalanced data training model use in balanced testing data")
p <- add_argument(p, "--input_csv", help="unbalanced data csv file",default = "../data/fake_job_postings.csv")
p <- add_argument(p, "--model_weight", help="training model weight",default = "../model_results/unb/gbm_ub.rds")
p <- add_argument(p, "--output_csv", help="evaluation table",default = "../model_results/unb/cnf_gbm_unb.csv")
# trailingOnly 如果是TRUE的話，會只編輯command-line出現args的值args <- 
args <- parse_args(p, commandArgs(trailingOnly = TRUE))

df1<-read.csv(args$input_csv,fileEncoding='utf-8')
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

summary(my_model_gbm,cBars=10)
gbm.perf(my_model_gbm)
pred_gbm <- predict(object=my_model_gbm,newdata=test,type="response")
predbin_gbm <- as.factor(ifelse(pred_gbm>0.5,1,0))
k <- as.factor(test$fraudulent)
confusematrix_gbm <- confusionMatrix(predbin_gbm,k)
acc.confusematrix_gbm <- confusematrix_gbm$overall[[1]]
sens.confusematrix_gbm <- confusematrix_gbm$byClass[[1]]
spec.confusematrix_gbm <- confusematrix_gbm$byClass[[2]]
prec.confusematrix_gbm <- confusematrix_gbm$byClass[[5]]
rec.confusematrix_gbm <- confusematrix_gbm$byClass[[6]]
f1.confusematrix_gbm <- confusematrix_gbm$byClass[[7]]
balacc.confusematrix_gbm <- confusematrix_gbm$byClass[[11]]
cgbm <- data.frame(round(acc.confusematrix_gbm,2),round(sens.confusematrix_gbm,2),round(spec.confusematrix_gbm,2),
                   round(prec.confusematrix_gbm,2),round(rec.confusematrix_gbm,2),round(f1.confusematrix_gbm,2),
                   round(balacc.confusematrix_gbm,2))
names(cgbm) <- c("accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")
write.csv(cgbm,file=args$output_csv)
roc.gbm <- rocit(pred_gbm, test$fraudulent)
summary(roc.gbm)
png(filename=paste("gbm_unb",".png",sep=""))
plot(roc.gbm, YIndex = F, values = F)
dev.off()

