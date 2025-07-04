# Language Detection
-----
Detection of the language of a text with the Multinomial Naive Bayes method and Neural Network Model 

# Introduction 
Certain tasks in natural language processing (NLP)—such as **automatic machine translation**, **multilingual information retrieval**, and **cross-lingual sentiment analysis**—require accurate identification of the input text's language. Language identification is a well-studied problem, with various approaches ranging from probabilistic models such as **Multinomial Naive Bayes**, to distance-based classifiers like **k-Nearest Neighbors**, and more recently, **neural network-based** methods.
This repository presents a comparative implementation of two such methods: a Multinomial Naive Bayes classifier as a statistical baseline, and a two-layer feedforward neural network that leverages character-level n-gram features to improve classification accuracy. 

> If you're interested in language detection at the word level rather than for entire text segments, I've also implemented a specialized solution for that. It's available in the [Detect-Code-Switching](https://github.com/javadr/PyTorch-Detect-Code-Switching) repository, which includes a [Streamlit-based web app](https://pytorch-detect-code-switching-code.streamlit.app/) for testing the model interactively.

# Installing dependencies
You can use the `pip` program to install the dependencies on your own. They are all listed in the `requirements.txt` file.

To use this method, you would proceed as follows:

```
pip install -r requirements.txt
```

# The Methods
For feature extraction, a **Count Vectorizer** is used to generate **character-level bi-grams and tri-grams**, 
which effectively capture language-specific subword patterns. 
To reduce noise and improve generalization, only n-grams with a minimum document frequency of `5` are included.

Both models are trained on these features:
- The **Multinomial Naive Bayes classifier** achieves an accuracy of approximately **96%** on an external test set.
- A **two-layer feedforward neural network** reaches a comparable performance level, also achieving around **96%** accuracy.
  
# Datasets
Two completely separate datasets are used for the training/validation and testing phases. 
The two following links are used for the mentioned phases, respectively. The first dataset has been modified to include Persian.

* [training/validation phase] `https://www.kaggle.com/basilb2s/language-detection`
* [testing phase] `https://ufal.mff.cuni.cz/~kocmanek/2016/docs/lanidenn_testset.txt`


# 📘 How to Use This Codebase
## 🚀 Launching the Streamlit App

To start the interactive app, run the following command from the root directory of the project:

```bash
streamlit run src/app.py
```

This will launch a Streamlit-based web interface that allows you to interactively test the trained model for language detection on custom text inputs.

If you're only interested in training and evaluating the model without using the web interface, you can skip this step and proceed directly to training.
## 🧠 Training the Neural Network

To train the language detection model using the neural network implementation, run:

```bash
python src/train_nn.py
```

This script will handle data loading, model training, and output performance metrics after completion.