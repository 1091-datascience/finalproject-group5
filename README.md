# [Group5] [Real or Fake] Fake JobPosting Prediction

### Groups
* 朱家宏, 106305036
* 林宗霖, 109753125
* 林瀚陞, 107751005
* 黃瑜萍, 108258021

### Goal
在求職階段時或是找實習時，不想因為假消息而讓自己在無意義的事浪費時間

### Demo 
在 finalproject-group5 的資料夾下執行

* 將 Unbalanced Data 轉成 Balanced Data
```R
Rscript ./code/Unbalanced_data2Balanced_data --input ./data/fake_job_postings_TFIDF.csv 
--output ./data/fake_job_postings_TFIDF_balance.csv
```

* 訓練GBM模型在 Unbalanced data 並評估模型好壞
```R
Rscript ./code/Unb_training_model_and_evl_table.R --input ./data/fake_job_postings.csv 
--output ./model_results/unb/cnf_gbm_unb.csv --training_rds ./model_results/unb/gbm_ub.rds
```

* 訓練 Decision_tree\GBM\xgboost\Lasso\Ridge 模型並評估模型好壞
```R
Rscript ./code/****.R --input ./data/fake_job_postings_TFIDF_balance.csv 

註1:****.R 可以替換成在code資料夾底下的模型名稱(e.g. decision_tree.R)

註2:還有其他的 arg_parser 分別代表

--training_rds 存放訓練Training data後的模型參數位置
--training_and_val_rds 存放訓練Training data + Validation data後的模型參數位置
--val_eval_table 存放訓練Training data後的模型評估指標csv檔
--testing_eval_table 存放訓練Training data + Validation data後的模型評估指標csv檔
--val_ROC 存放訓練Training data後的模型ROC的png檔
--testing_ROC 存放訓練Training data + Validation data後的模型ROC的png檔
--fold 需要將資料集切的份數
--training_cv_rds 找出Training data後的模型最佳的lambda值(這是Ridge/Lasso才有)
--training_and_val_cv_rds 找出Training data + Validation data後的模型最佳的lambda值(這是Ridge/Lasso才有)

以上arg_parser你可以直接使用內建的預設值
```

P.S. 我們訓練好的模型 Weights 我們有另外放在雲端，請自行下載

(https://drive.google.com/drive/folders/1HDY8g8NNHdUHut-sOkXCeM_MjAkRTAxR?usp=sharing)

* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1091_datascience_FP_<yourID|groupName>.ppt/pptx/pdf, by **Jan. 12**
* Any related document for the final project
  * papers
  * software user guide

### data

* Kaggle 公開資料 Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
* 原始資料來自希臘愛琴大學 資安實驗室的 EMSCAD 資料集
 ![Kaggle公開資料](/Images/EMSCAD資料集.png)

* Input format
* 根據官方原始文件
 ![Input_format](/Images/Input_format.png)
* Any preprocessing?

  * Handle missing data   
    * 因為 Salary 太多 Missing Value，所以將此欄位的資料刪除
    * 將 Title / Department 刪除，因為 Desciption 欄位會描述
    
  * Scale value 
    * 將類別型的資料轉成 One-Hot-Encoding 形式
    
  * 將文字型資料合併成一欄並使用 TF-IDF 技術挑選文本中比較重要的文字，並取其前大約4000大的值，之後在將有這些文字的資料做 One-Hot-Encoding
  
    (R檔跑不了這麼大量的資料，所以我們使用python來做)
    
    連結：https://colab.research.google.com/drive/1W6wSqikmaq6s2pE6yhDOoYl4I9jugM18?usp=sharing)
      
  * Unbalanced data preprocessing 
    * 在 Label 數量極為不平均，真工作假工作比例大約為 10:1，所以我們利用 Synthetic Minority Oversampling Technique(SMOTE)方法減少真工作資料以及增加假工作資料，使其比例接近 1:1
  
### code

* Which method do you use? 
  * 我們針對這些資料分別使用下列模型 : gbm/decision_tree/lasso/ridge/xgboost
* What is a null model for comparison? 
  * null_model : 全部資料都預測為真工作(因為在資料中真工作比例很高)
* How do your perform evaluation? 
  * 利用 10-fold 交叉驗證

### results

* Which metric do you use 
  * accuracy/sensitivity/specificity/precision/recall/F1-score/balanced_accuracy/AUC
* Is your improvement significant?
  * 先用訓練好的模型各自與原先的null_model以balanced_accuracy做比較，再將全部的模型做第二階段以同樣方式做篩選，選出在這資料集中最佳的模型
* What is the challenge part of your project?
  * 在這挑戰中，因資料集有文字型且假工作與真工作的比例極為不平均需針對這些資料作特別的前處理
## References
* Code/implementation which you include/reference

* Packages you use

  *  library(psych)  
  *  library(ggplot2) 
  *  library(caret)
  *  library(ROCit)
  *  library(argparser)
  *  library(gbm)
  *  library(glmnet)
  *  library(unbalanced)
  *  library(xgboost)
  *  library(rpart)

* Related publications
  * paper
    * SMOTE: Synthetic Minority Over-sampling Technique(https://arxiv.org/pdf/1106.1813.pdf) 

