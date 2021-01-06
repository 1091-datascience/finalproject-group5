library(unbalanced)
library(argparser)

p <- arg_parser("Process unbalanced data csv to balanced data csv")#
p <- add_argument(p, "--input_csv", help="unbalanced data csv file",default = "./data/fake_job_postings_TFIDF.csv" )
p <- add_argument(p, "--output_csv", help="balanced data csv file",default = "./data/fake_job_postings_TFIDF_balance.csv")
# trailingOnly 如果是TRUE的話，會只編輯command-line出現args的值args <- 
args <- parse_args(p, commandArgs(trailingOnly = TRUE))

df1<-read.csv(args$input_csv,fileEncoding='utf-8')
y <- as.factor(df1$fraudulent)
x <- df1[,c(-1,-5)]
# (Synthetic Minority Oversampling Technique) 
data_osp <- ubBalance(X=x, Y=y, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
balancedData_x<-data_osp$X
fraudulent <- data_osp$Y
balanceData <- cbind(balancedData_x,fraudulent)
write.csv(balanceData,file=args$output_csv)
