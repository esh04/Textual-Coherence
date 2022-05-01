# Textual-Coherence

## Dataset

The datasets used are:
1. The Grammarly Corpus of Discourse Coherence


The dataset is annotated into classes 1/2/3 where 3 denotes the most coherent paragraph. 
Instructions regarding the dataset can be found [here](https://github.com/aylai/GCDC-corpus)

2. Wikipedia/CNN Corpus


The dataset consists of a set of coherent sentences along with respective replacements to make them incoherent. 
Additional details can be found [here](https://github.com/AiliAili/Coherence_Modelling)

## Repository Structure
- The data is present in the data folder
- The code used to preprocess the data is present in the preprocessing folder
- We have used three models, each one is represented in respective notebooks. (LSTM.ipynb, GRU.ipynb, RNN.ipynb)

## How to Run?
- The datasets can be downloaded from the above given links. Preprocessed zip folders are present in the data folder of the repository.
- Steps to run the model are present in the respective notebooks.

## Approaches used


The following approaches were used on the GCDC corpus to fine tune the models: 

- Some of the testing data was used for training as we realised that training data was insufficient for an accurate prediction.
- The corpus had data annotated to 1/2/3 depicting the levels of coherence, so we initially implemented a three way classifier. Later we converted it into a binary classifier.This lead to a considerable increase in the accurace.
- We used cosine similarity between adjacent sentences of the paragraphs as a parameter to calculate coherence. At first we used average similarity of the paragraph which we then changed to minimum similarity. 

The above methods were tried on different training and testing datas from the GCDC corpus and the best model was saved.

The best model in each method was used on the **Wikipedia-CNN Corpus** to observe the accuracies. 

## Accuracies Observed

### Method 1 - LSTM
#### GCDC Corpus
- Without using similarity as a parameter
  - 3000 training data, 600 testing data, three way classifier: approx **30%**
  - 4600 training data, Yahoo test data, three way classifier: approx **36.5%**
  - 4600 training data, Yahoo test data, binary classifier: approx **55%**
  - 4600 training data, Clinton test data, binary classifier: approx **64.99%**
 
- Using Average Similarity as a parameter
  - 4600 training data, Clinton test data, three way classifier: approx **34%**

- Using Minimum Similarity as a parameter
  - 4600 training data, Clinton test data, three way classifier: approx **39.5%**
  - 4600 training data, Clinton test data, binary classifier: approx **67%**

- We observe different results with different test datas, thus we compared the performance of all the test data available in the GCDC corpus using that LSTM model, binary classification with minimum similarity as a parameter.
  - Results Obtained:
    - Clinton: approx **61%**
    - Enron: approx **66%**
    - Yahoo: approx **54.5%**
    - Yelp: approx **66.5%**
  - It is fair to assume that Enron has best performance as it is more closed domained than the rest. Similarly, the Yahoo Question-Answer corpus is the most open domained. 

Binary classifier performed significantly better that three way multi classifier, thus we used only binary classifiers for the Wikipedia-CNN corpus.


#### Wikipedia-CNN Corpus
- Using binary classifier without any similarity parameter
  - **71.66%**
- Using binary classifier with minimum similarity parameter
  - **74.55%**


### Method 2 - GRU
#### GCDC Corpus
- Without using similarity as a parameter
  - 4600 training data, 200 Clinton test data, three way classifier: approx **41.99%**
  - 4600 training data, 200 Clinton test data, binary classifier: approx **57.99%**
- Using Minimum Similarity as a parameter
  - 4600 training data, 200 Clinton test data, binary classifier: approx **63.99%**
#### Wikipedia-CNN Corpus

### Method 3 - RNN
#### GCDC Corpus
- Without using similarity as a parameter
  - 4600 training data, 200 Clinton test data, three way classifier: approx **32.49%**
  - 4600 training data, 200 Clinton test data, binary classifier: approx **55.5%**
- Using Minimum Similarity as a parameter
  - 4600 training data, 200 Clinton test data, binary classifier: approx **53.5%**
#### Wikipedia-CNN Corpus


**In depth analysis can be found in the Report.pdf**
