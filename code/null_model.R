library(psych)  #for general functions
library(argparser)
library(caret)

main_dir <- '../model_results'
sub_dir <- 'null_model'
output_dir <- file.path(main_dir, sub_dir)

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("Decision tree Dir already exists!")
}

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input_unb_csv", help="unbalanced data csv file",default = "../data/fake_job_postings.csv" )
p <- add_argument(p, "--input_bal_csv", help="balanced data csv file",default = "../data/fake_job_postings_TFIDF_balance.csv" )
p <- add_argument(p, "--eval_table", help="training",default = "../model_results/null_model/cnf_null_model.csv" )

# trailingOnly 如果是TRUE的話，會只編輯command-line出現args的值args <- 
args <- parse_args(p, commandArgs(trailingOnly = TRUE))

#df1 <- read.csv(args$input_unb_csv)
#df2 <- read.csv(args$input_bal_csv)

#unbalanced data
df1 <- read.csv(args$input_unb_csv)
ans <- df1$fraudulent
guess <- df1$fraudulent
guess <- ifelse(guess==1,0,0)
as.factor(ans)

result <- confusionMatrix(as.factor(guess),as.factor(ans),positive = "1")
acc <- result$overall[[1]]
sens <- result$byClass[[1]]
spec <- result$byClass[[2]]
prec <- result$byClass[[5]]
rec <- result$byClass[[6]]
f1 <- result$byClass[[7]]
balacc <- result$byClass[[11]]

csv <- data.frame('unbalanced_data',round(acc,4),sens,spec,prec,rec,f1,balacc)
names(csv ) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")

#balanced data

df2 <- read.csv(args$input_bal_csv,check.names = F)
ans <- df2$fraudulent
guess <- df2$fraudulent
guess <- ifelse(guess==1,0,0)
as.factor(ans)

result <- confusionMatrix(as.factor(guess),as.factor(ans),positive = "1")
acc <- result$overall[[1]]
sens <- result$byClass[[1]]
spec <- result$byClass[[2]]
prec <- result$byClass[[5]]
rec <- result$byClass[[6]]
f1 <- result$byClass[[7]]
balacc <- result$byClass[[11]]

csv_2 <- data.frame('balanced_data',round(acc,4),sens,spec,prec,rec,f1,balacc)
names(csv_2) <- c("set","accuracy","sensitivity","specificity","precision","recall","F1-score","balanced_accuaracy")

write.csv(rbind(csv,csv_2),file=args$eval_table)

