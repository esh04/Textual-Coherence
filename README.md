# Textual-Coherence

## Dataset

The datasets used are:
1. The Grammarly Corpus of Discourse Coherence to train our model. Instructions to use the same can be found [here](https://github.com/aylai/GCDC-corpus)

The dataset is annotated into classes 1/2/3 where 3 denotes the most coherent paragraph. 

2. https://github.com/AiliAili/Coherence_Modelling

The dataset consists of a set of coherent sentences along with respective replacements to make them incoherent. 

## Repository Structure
- The data is present in the data folder
- The code used to preprocess the data is present in the preprocessing folder
- We have used three models, each one is represented in respective notebooks. (LSTM.ipynb, GRU.ipynb, RNN.ipynb)

## How to Run?
- The datasets can be downloaded from the above given links. Preprocessed zip folders are present in the data folder of the repository.
- Steps to run the model are present in the respective notebooks.

## Approaches used


The following approaches were used on the GCDC corpus to fine tune the models: 

- The corpus had data annotated to 1/2/3 depicting the levels of coherence, so we initially implemented a three way classifier. Later we converted it into a binary classifier.
- We used cosine similarity between adjacent sentences of the paragraphs as a parameter to calculate coherence. At first we used average similarity of the paragraph which we then changed to minimum similarity. 

The above methods were tried on different training and testing datas from the GCDC corpus and the best model was saved.

The best model in each method was used on the Git Corpus to observe the accuracies. 

## Accuracies Observed

### Method 1 - LSTM

### Method 2 - GRU

### Method 3 - RNN


**In depth analysis can be found in the Report.pdf**
