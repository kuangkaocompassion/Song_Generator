## DATA SET 
* Emoji data set   
* NLPCC data set    
## RUNNING STEPS

#### NLPCC
* Extract All Sentences and Emotion Tag from files    
"Training_data_for_Emotion_Classification.xml"    
"Training_data_for_Emotion_Expression_Identification.xml"  
```shell
$ python3 Weibo_Data_Process.py
```
* Tokenized Sentences  
```shell
$ python3 Data_Process.py
```
* Training Emotion Classification Model  
```shell
$ python3 Emotion_Classification.py
```
* Experiment Record  
Experiment_Record.csv  
