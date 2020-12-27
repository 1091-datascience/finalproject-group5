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
You should provide an example commend to reproduce your result
```R
Rscript code/your_script.R --input data/training --output results/performance.tsv
```
* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1091_datascience_FP_<yourID|groupName>.ppt/pptx/pdf, by **Jan. 12**
* Any related document for the final project
  * papers
  * software user guide

### data

* Kaggle 公開資料 Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
![Kaggle公開資料](/Images/EMSCAD資料集.png)

* Input format
* Any preprocessing?
  * 將文字型資料合併成一欄在使用TF-IDF技術挑選文本中比較重要的文字
  * Handle missing data   *將Salary 資料刪除(因為太多missing value) 將title刪除，因為desciption欄位會描述
  * Scale value           *將類別型的資料轉成one-hot-encoding形式
  * Unbalanced data preprocessing 要補
  
### code

* Which method do you use? *Decision_Tree ridge lasso gbm xgboost
* What is a null model for comparison? *Unbalanced model using in balanced data
* How do your perform evaluation? ie. Cross-validation, or extra separated data
P.S. 我們訓練好的模型weight我們有另外放在雲端，請自行下載(https://drive.google.com/drive/folders/1WpWAUbflBEZDdUu03k2wPS2FTHHsYLf7?usp=sharing)

### results

* Which metric do you use 
  * precision, recall, R-square
* Is your improvement significant?
* What is the challenge part of your project?

## References
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)
* Packages you use
* Related publications


