# Machine learning algorithms
A collection of minimal and clean implementations of machine learning algorithms.

### Why?
This project is targeting people who want to learn internals of ml algorithms or implement them from scratch.  
The code is much easier to follow than the optimized libraries and easier to play with.  
All algorithms are implemented in Python, using numpy, scipy and autograd.  

### Implemented:
* [Deep learning (MLP, CNN, RNN, LSTM)](mla/neuralnet)
* [Linear regression, logistic regression](mla/linear_models.py)
* [Random Forests](mla/ensemble/random_forest.py)
* [Support vector machine (SVM) with kernels (Linear, Poly, RBF)](mla/svm)
* [K-Means](mla/kmeans.py)
* [Gaussian Mixture Model](mla/gaussian_mixture.py)
* [K-nearest neighbors](mla/knn.py)
* [Naive bayes](mla/naive_bayes.py)
* [Principal component analysis (PCA)](mla/pca.py)
* [Factorization machines](mla/fm.py)
* [Restricted Boltzmann machine (RBM)](mla/rbm.py)
* [t-Distributed Stochastic Neighbor Embedding (t-SNE)](mla/tsne.py)
* [Gradient Boosting trees (also known as GBDT, GBRT, GBM, XGBoost)](mla/ensemble/gbm.py)
* [Reinforcement learning (Deep Q learning)](mla/rl)


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
Feel free to improve existing code, documentation or implement new algorithm.  
Please open an issue to propose your changes if they are big enough.  
