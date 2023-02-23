# PRINT
PRINT: Personalized Relevance Incentive Network for CTR Prediction in Sponsored Search

This article only provides code for public datasets and paper methods, the data preprocessing script is in the script directory，
and main.py is the execution entry of our method
##Data Preprocessing
we download the KDD2012 Dataset from https://www.kaggle.com/competitions/kddcup2012-track2/data。

We randomly choose to retain 10% of users to improve the computation efficiency. Then 50% of the sampled samples are used to make user interaction sequences, 40% are used as training set, and 10% are used as validation set

## BERT MODEL
We choose the open source BERT model to process public datasets, and simplify the tokenization of text sequences 
https://huggingface.co/prajjwal1/bert-small


