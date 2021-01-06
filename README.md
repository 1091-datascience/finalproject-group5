# [Group5] [Real or Fake] Fake JobPosting Prediction

### Groups
* 朱家宏, 106305036
* 林宗霖, 109753125
* 林瀚陞, 107751005
* 黃瑜萍, 108258021

### Goal
A breif introduction about your project, i.e., what is your goal?

This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.

### Demo 
在finalproject-group5的資料夾下執行

將Unblanced Data 轉成Balanced Data
```R
Rscript ./code/Unbalanced_data2Balanced_data --input_csv ./data/fake_job_postings_TFIDF.csv 
--output_csv ./data/fake_job_postings_TFIDF_balance.csv
```

訓練GBM模型在Unbalanced data並評估模型好壞
```R
Rscript ./code/Unb_training_model_and_evl_table.R --input_csv ./data/fake_job_postings.csv 
--output_csv ./model_results/unb/cnf_gbm_unb.csv --model_weight ./model_results/unb/gbm_ub.rds
```

訓練Decision_tree\GBM\xgboost\Lasso\Ridge 模型並評估模型好壞
```R
Rscript ./code/****.R --input ./data/fake_job_postings_TFIDF_balance.csv 

註1:****.R 可以替換成在code資料夾底下的模型名稱(e.g. decision_tree.R)

註2:還有其他的arg_parser 分別代表

--training_rds 存放訓練Training data後的模型參數位置
--training_and_val_rd 存放訓練Training data + Validation data後的模型參數位置
--val_eval_table 存放訓練Training data後的模型評估指標csv檔
--testing_eval_table 存放訓練Training data + Validation data後的模型評估指標csv檔
--val_ROC 存放訓練Training data後的模型ROC的png檔
--testing_ROC 存放訓練Training data + Validation data後的模型ROC的png檔
--training_cv_rds (這是Ridge/Lasso才有)
--training_and_val_cv_rds (這是Ridge/Lasso才有)
--val_eval_table (這是Ridge/Lasso才有)
--testing_eval_table (這是Ridge/Lasso才有)

以上arg_parser你可以直接使用內建的預設值
```

P.S. 我們訓練好的模型Weights我們有另外放在雲端，請自行下載
# (https://drive.google.com/drive/folders/1HDY8g8NNHdUHut-sOkXCeM_MjAkRTAxR?usp=sharing)

* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1091_datascience_FP_<yourID|groupName>.ppt/pptx/pdf, by **Jan. 12**
* Any related document for the final project
  * papers
  * software user guide

### data

* Kaggle 公開資料 Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
* 原始資料來自希臘愛琴大學 資安實驗室的EMSCAD資料集
 ![Kaggle公開資料](/Images/EMSCAD資料集.png)

* Input format
* 根據官方原始文件
 ![Input_format](/Images/Input_format.png)
* Any preprocessing?
  * 將文字型資料合併成一欄在使用TF-IDF技術挑選文本中比較重要的文字 (因為R檔跑不了這麼大量的資料，所以我們使用python來做連結：https://colab.research.google.com/drive/1W6wSqikmaq6s2pE6yhDOoYl4I9jugM18?usp=sharing)
  * Handle missing data   *將Salary 資料刪除(因為太多missing value) 將title刪除，因為desciption欄位會描述
  * Scale value           *將類別型的資料轉成one-hot-encoding形式
  * Unbalanced data preprocessing *在label數量極為不平均大約為10:1 利用Synthetic Minority Oversampling Technique(SMOTE)方法合成少數類
  
### code

* Which method do you use? *Decision_Tree ridge lasso gbm xgboost
* What is a null model for comparison? *Unbalanced model using in balanced data
* How do your perform evaluation? ie. Cross-validation, or extra separated data

### results

* Which metric do you use 
  * precision, recall, R-square
* Is your improvement significant?
* What is the challenge part of your project?

## References
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)
* Packages you use
* Related publications


