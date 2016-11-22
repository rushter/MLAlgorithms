# Machine learning algorithms
A collection of minimal and clean implementations of machine learning algorithms.

### Why?
This project is targeting people who want to learn internals of ml algorithms or implement them from scratch.  
The code is much easier to follow than the optimized libraries and easier to play with.  
All algorithms are implemented in Python, using numpy, scipy and autograd.  

### Implemented:
* [Deep learning (MLP, CNN, RNN, LSTM)] (mla/neuralnet)
* [Linear regression, logistic regression] (mla/linear_models.py)
* [Random Forests] (mla/ensemble/random_forest.py)
* [SVM with kernels (Linear, Poly, RBF)] (mla/svm)
* [K-Means] (mla/kmeans.py)
* [K-nearest neighbors] (mla/knn.py)
* [Naive bayes] (mla/naive_bayes.py)
* [PCA] (mla/pca.py)
* [Factorization machines] (mla/fm.py)
* [Gradient Boosting trees (also known as GBDT, GBRT, GBM, XGBoost)] (mla/ensemble/gbm.py)


### TODO:
* t-SNE
* MCMC
* Word2vec
* Adaboost
* HMM
* Restricted Boltzmann machine

### Installation
        git clone https://github.com/rushter/MLAlgorithms
        cd MLAlgorithms
        pip install scipy numpy
        pip install .

### How to run examples without installation
        cd MLAlgorithms
        python -m examples.linear_models

### How to run examples within Docker
        cd MLAlgorithms
        docker build -t mlalgorithms .
        docker run --rm -it mlalgorithms bash
        python -m examples.linear_models

### Contributing
Your contributions are always welcome!
