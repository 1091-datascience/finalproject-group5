df <- read.csv("fake_job_postings.csv", header = T, stringsAsFactors = F, sep=",")
df <- df[1:18]
counter <- sapply(df, function(x){length(unique(x))})
dT <- as.data.frame(counter)
colnames(dT) <- c("Varibales")
dT1 <- t(dT)

print(dT1)
