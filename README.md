# PRIMA

## source code for the paper : A Tale of Two Tasks: Automated Issue Priority Prediction with Deep Multi-task Learning .

### Requirement

1. python 3.7  
2. pytorch 1.1

### How to run?

1. You should download "pytorch_model.bin" file ( a pre-trained Bert model )  from https://huggingface.co/lanwuwei/BERTOverflow_stackoverflow_github/tree/main, and put it into the " bert_pretrain " folder of this project

2. Run "**python run.py --model muti_bert_rcnn_att**" to  train  Prima (default: kubernetes dataset)

   

3. *if you want to change dataset,  modify the first code line of **run.py**.*

4.  *if you want to change model and mode, you will need to change  **run.py** and the commend.*

   *For example,* 

   *change  "from muti_train_eval import train" to "from priority_train_eval import train" in  **run.py** ,*

   *meanwhile, Run "**python run.py --model priority_bert_rcnn_att**" to  train  single priority task of PRIMA.* 


