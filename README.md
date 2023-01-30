# Overview
- We are provided with over 1 million texts which are annotated as negative, positive and neutral. The task
is to create features based on the provided data and use machine learning algorithms to classify them.
- We used different strategies to build the features, such as n-gram (bag of words, bi-gram, tri-gram),
TF-IDF and word embedding(word2vec). Similarly, we also tried different models to implement classifiers
such as Naive Bayes, Support Vector Machines, and Neural Networks (NN and RNN). In terms of results,
the best score for Kaggle using the method SVM is 0.82391

# contents
- Readme.md                   // help
- dataset  // dataset from the kaggle <--need additional download   
- NaiveBayes.ipynb            // Naive Bayes using Bag of words and TF-IDF features
    - naivebayes.py
- SVM.ipynb                   // Kernelized SVM
    - svm.py
- NNandRNN.ipynb              // NN and RNN using Bag of words features
    - nn.py
- TextCNN.ipynb               // TextCNN using word2vec features
    - DataPrepare.ipynb       // Preprocessing for TextCNN
    - TrainW2V.ipynb          // Train word2vec
    - textcnn.py              
    - trainw2v.py
- nb-with-explainability.ipynb      //  explainability of Naive Bayes
    - nb-with-explainability.py

# How to run
step1:
Download the dataset from the kaggle website and put the folder in the correct location as shown in the directory
    Text Classification Challenge:
    https://www.kaggle.com/t/6c98b08e131d49abaa8915175de22ced/data

step2:
Configure the right environment, for example: Colab or JupyterNotebook
Note: The pytorch version is 1.13.0

step3:
To reproduce the project, run the following notebooks follow the annotations:
- `NaiveBayes.ipynb`
- `SVM.ipynb` 
- `NNandRNN.ipynb` 
- `nb-with-explainability.ipynb` 
To reproduce the TextCNN, run the following notebooks in the given order and follow the annotations: 
- `DataPrepare.ipynb` 
- `TrainW2V.ipynb`
- `TextCNN.ipynb`