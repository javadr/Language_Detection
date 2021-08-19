# Language Detection 
-----
Detection of the language of a text with Multinomial Naive Bayes method and Neural Network Model 

# Introduction 
Some tasks in NLP, for instance automatic machine translation, need to know about the language of the input text. There are lot of methods that can solve this issue, like **Naive Bayes**, **k Nearest Neighbours**, **Kulback Leibler Divergence**, and some others. 
This python notebook tries to find the language of the input text. It uses Multinomial Naive Bayes method as a baseline and a two layer neural network in order to solve it. 

# Installing dependencies
You can use the `pip` program to install the dependencies on your own. They are all listed in the `requirements.txt` file.

To use this method, you would proceed as:

```
pip install -r requirements.txt
```

# The Methods
As a feature selection, the Count Vectorizer have been used to count the bi-grams and tir-grams with minimum frequency of `5`. 
The Naive Bayes achieve got `96%` accuracy on an unseen data set while the trained neural network approached the `97%`. 

# Datasets
Two completely separate datasets are used for training/validation and testing phase. 
The two following links are used for the mentioned phases, respectively. The first dataset has been modified to include Persian. 

* [training/validation phase] `https://www.kaggle.com/basilb2s/language-detection`
* [testing phase] `https://ufal.mff.cuni.cz/~kocmanek/2016/docs/lanidenn_testset.txt`
