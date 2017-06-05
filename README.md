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
$ python3 Model_Data_process.py (training_csv_name)
$ python3 Model_Emotion_Classification_bilstm.py TRAIN (train_model) (epoch)
```
* Using Emotion Classification Model
```shell
$ python3 USE_Data_process.py (using_csv_name)
$ python3 Model_Emotion_Classification_bilstm.py USE (used_model) (result_csv_name) 
```
* Finetuning Emotion Classification Model 
```shell
$ python3 Finetune_Data_process.py (finetuning_csv_name)
$ python3 Model_Emotion_Classification_bilstm.py FINETUNE (original_model) (epoch)
```
* Experiment Record  
Experiment_Record.csv  
