---
layout : post
title: "Why generalizing beyond training data is so difficult?"
excerpt: "After a hiatus, I am back to jot down some notes on why machine learning algorithms struggle with generalizing beyond the training data."
category : posts
last_modified_at: 
tags: 
  - Python
  - Machine Learning
published: true
date: 2019-05-03
---

I recently came across a machine learning review [_A high-bias, low-variance introduction to Machine Learning for physicists_](http://physics.bu.edu/~pankajm/MLnotebooks.html). I found it a pleasure to read for several reasons. It is really well-written and I find it quite accessible. It also comes with a set of great jupyter notebooks that illustrate key ideas through python code that readers can play with and do some exercises. These notebooks are available [online](http://physics.bu.edu/~pankajm/MLnotebooks.html). The review and corresponding notebooks are packed with great material. I decided to read it and comprehend as much as of it as possible by reproducing the notebooks. In this manner, I will be able to collect my notes and get practice the main ideas through coding. So here it goes, the first post on the topic of generalization in machine learning and why it is so hard.

# Setting the stage

The setup for a typical Data Science or Machine Learning problem is as follows. We have a dataset, $$\mathbf{X}$$. We wish to fit a model $$g(\mathbf{w})$$, which is a function of paramaters $$\mathbf{w}$$. The process of fitting amounts to finding values of $$\mathbf{w}$$ that minimize the cost function, $$C(\mathbf{X}, g(\mathbf{w}))$$. A key step in this approach is to split the data into two subsets, a training set $$\mathbf{X}_{train}$$ and a test set $$\mathbf{X}_{test}$$. 


The training set is used for model fitting and therefore  model parameters are given by: 

$$\;{\tiny\begin{matrix}
\\ \normalsize argmin 
\\ ^{\scriptsize \mathbf{w}}\end{matrix} }\;
\{C(\mathbf{X}_{train}, g(\mathbf{w}))\}
$$

The value of the cost function for the training set $$C(\mathbf{X}_{train}, g(\mathbf{w}))$$  is called the training error or in-sample error, $$E_{in}$$. The value of the cost function for the test set $$C(\mathbf{X}_{train}, g(\mathbf{w}))$$ is called the test error or out-of-sample error, $$E_{out}$$. In practice,  $$\mathbf{X}_{train}$$ is actually dvided into $$\mathbf{X}_{train}$$ and $$\mathbf{X}_{validation}$$, where the former is used for fitting and the later is used for hyperparameter tuning. We will ignore $$\mathbf{X}_{validation}$$ for now. The out-of-sample error $$E_{out}$$ is an unbiased measure of model performance, provided that test set is kept completely separate during model building, including steps involving feature scaling and selection. In most cases, we are unaware of the mathematical model that describes the data, and, therefore we make some assumptions and try a bunch of different models. The model that gives lowest $$E_{out}$$ is selected and this is where the challenege begins. It turns out that the model with the lowest $$E_{out}$$ is not necessarilty the one with the lowest $$E_{in}$$. Indeed, the model with the lowest $$E_{in}$$ most likely has a poor performance on the the test set, because it is probably overfitting the training data. 



This gap between $$E_{in}$$ and $$E_{out}$$ is at the heart of a major difficulty in machine learning. i.e., building models that generalize well to examples outside of training data. Here we will focus on some of the factors that influence this gap, namely the size of training data, noise in the measurement of training data and complexity of the underlying model. We will see that the noise leads to unreal trends in training data and when models are fit to such trends, the performance over test sets reduces significantly. These effects are even more pronounced with small amount of training data and/or with more complex models. An increase in the size of training data may drown out some of the noise and avoid ovefitting. Let's get started by coding up some useful functions that will help us understand these ideas.



## Generating Data

We start by coding up a function to generate the data. This function mimics a process that generates data with intrinsic noise. For the purpose of this exercise, we can control the level of noise. We will also add an option for generating data either from a linear model, $$f(x) = 2x$$ or from a non-linear model $$H$$, $$f(x) = 2x - 10x^5 + 15x^{10}$$. 

```python
%matplotlib inline
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.style.use("fivethirtyeight")
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
```



```python
def generate_data(x_interval, n_train, sigma=1.0, linear=True):
    """
    Generate data in a given range of input values,  
    
    Parameters
    ----------
    x_interval : tuple
        A tuple containing the range in which the input data is generated.
    n_train : int
        The number of training examples.
    sigma : float
        Noise strength, this is the standard deviation of the Guassian noise.
    linear : bool
        If True, return data from a linear function, otherwise, return 
        data from a non-linear function.
        
    Returns
    -------
    x : numpy.ndarray
        Input values in the range defined by x_interval
    y : numpy.ndarray
        Labels for the input data generated from the model
    """
    x_0, x_n_train = x_interval
    s = sigma * np.random.randn(n_train)
    x = np.linspace(x_0, x_n_train, n_train)

    if linear:
        y = (2 * x) + s
    else:
        y = (2 * x) - (10 * x**5) + (15 * x**10) + s
    return x[:, np.newaxis], y
```

## Building Models
Now we will code up a function that returns model(s) that are fit to the training data. The models returned from this function belong to the class of polynomial regression moodels. In order to use this function to generate polynomial regression models of degree > 1, we have to first transform our data prior to using this function so that it contains polynomial features. For this purpose, we also code up the transformation function that takes our one-dimensional data as input and produces the transformation with additional polynomial features for the given degree, e.g., for degree 2, the transformation with create a feature matrix containing the original variable $$x_i$$ and a new feature $$x_i^2$$ for $$i^{th}$$ training exmaple. As you can notice, the model fitting is performed through `scikit-learn` `LinearRegression` estimator using default arguments. 


```python
def build_model(data, y):
    """
    Returns a polynomial regression model of degree upto but not more than 10. 
    
    Parameters
    ----------
    data : numpy.ndarray, shape = (n_train, order)
        Training data, it's assumed that data is already transformed for polynomials of order >=1. 
    y : numpy.ndarray, shape(n_train,)
        Labels or target values.
    
    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        A scikit-learn linear regression model object fit on the input data using default settings.  
    """

    degree = data.shape[1] - 1
    if degree > 10:
        raise ValueError("Degree %d is not supported, try degree=<%d" % (degree, 10))
    else:
        model = linear_model.LinearRegression()
        model.fit(data, y)
        return model

def transform_data(data, degree):
    """
    Transforms input data for polynomials of degree >= 1.
    
    Parameters
    ----------
    data : numpy.ndarray, shape = (n_train, 1)
        Training data, in the shape of a column vector. 
    degree : int
        Order of the polynomial for which data needs to be transformed.
    
    Returns
    -------
    X : numpy.ndarray, shape = (n_train, degree)
        Transformed input data.
    """

    if degree == 0:
        return data
    else:
        poly_features = PolynomialFeatures(degree)
        X = poly_features.fit_transform(data)
        return X
```

## Analyzing Results
We will run a bunch of experiments so let's write a couple of utility functions that allow quick and easy comparisons. 


```python
def mean_squared_error(y, y_pred):
    """
    Returns mean squared error between target and predicted values.
    
    Parameters
    ----------
    y : numpy.ndarray
        Target values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns
    -------
    ase : float
        Mean squared error between target and predicted values
    """

    mse = np.mean((y - y_pred)**2)
    return mse


def run_experiment(n_train=10, n_test=10,
                   x_interval_train=(0.00, 1.00), x_interval_test=(0.00, 1.40),
                   linear="True", noise=0.0):
    """
    Returns results from a given experiment that specifies number and ranges of train and test data, 
    the type and level of noise in underlying data geenrating process. Each experiment is run with all
    three models under study.

    Parameters
    ==========
    n_train : int
        Number of training examples
    n_test : int
        Number of test examples
    x_interval_train : tuple
        The range of training data
    x_interval_test : tuple
        The range of test data
    noise : float
        Strength of noise in data generating porcess
    linear : bool
        If True the data generatiing process is linear, otherwise, non-linear.
    
    Returns
    =======
    inputs, results, errors : tuple
        inputs: list containing transformed train and test datasets
        results: a dictionary where each model class (linear, 3-degree or 10-degree polynonomial) 
        is a key and its value is a list of containing predictions on train and test data,e.g.,
        results["linear"] = [np.ndarry of training data predictions, np.ndarry of test data predictions].
        True labels are also stored under the key "target". 
        errors: a dictionary where each model class (linear, 3-degree or 10-degree polynonomial) 
        is a key and its value is a list containing errors on train and test data,e.g.,
        results["linear"] = [training error, test error].
    """

    model_names = ["linear", "poly3", "poly10"]
    model_degrees = [0, 3, 10]

    inputs = []
    results = {n: [] for n in model_names + ["target"]}
    errors = {n: [] for n in model_names}
    
    
    x_train, y_train = generate_data(x_interval_train, n_train, noise, linear=linear)
    x_test, y_test = generate_data(x_interval_test, n_test, noise, linear=linear)
    
    inputs.append(x_train)
    inputs.append(x_test)
    results["target"].append(y_train)
    results["target"].append(y_test)

    # generate training and test data
    for index, model_name in enumerate(model_names):
        degree = model_degrees[index]
        
        x_train_transform = transform_data(x_train, degree)
        x_test_transform = transform_data(x_test, degree)
        
        model = build_model(x_train_transform, y_train)
        y_train_pred = model.predict(x_train_transform)
        e_in = mean_squared_error(y_train, y_train_pred)
        y_test_pred = model.predict(x_test_transform)
        e_out = mean_squared_error(y_test, y_test_pred)
        
        errors[model_name].append(e_in)
        errors[model_name].append(e_out)

        results[model_name].append(y_train_pred)
        results[model_name].append(y_test_pred)
        
    return inputs, results, errors


def plot_results(inputs, results, ylim=(-4.0, 4.0), yticks=5):
    """
    Generate plots of train and test data along with fitted functions. 
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex='col')
    labels = ["Training", "Test"]
    
    for model in results.keys():
        model_name = model
        for index, ax in enumerate(axes):
            x, y = inputs[index], results[model_name][index]
            marker = "-"
            marker_size = 16
            if model_name == "target":
                marker = "o"
                marker_size = 8
            ax.plot(x, y, marker, ms=marker_size, alpha=1.0, label=model_name)

    for index, ax in enumerate(axes):
        ax.legend()
        ax.set_ylim(ylim[0], ylim[1])
        ax.yaxis.set_major_locator(plt.MaxNLocator(yticks))
        ax.set_title(labels[index])
```

Please note that the function `run_experiment` hides away a lot of under-the-hood stuff so that we can focus on the results here. Just to refresh, it works by first generating some data from either linear or non-linear class and with or without the noise (depending on the arguments supplied at the call). Then it uses three models (linear, a degree 3 poylnomial and a degree 10 poylnomial). This is followed by testing the models on a separate test set. For each model, the training and test set errors are also recorded. The `plot_results` takes care of the plotting. All of this is meant to mimic what happens in a typical machine learning project; i.e.
* We have data with features and target variables, it is split into training and test sets.
* Since we don't know the true mathematical relationship, therefore, we chose a set of models to fit on the training data and then we use error on the test set to choose the best model.

Now that everything is ready, let's run some experiments. 

### Experiment 1
* No noise in the data, $$\sigma = 0$$
* Small training dataset, $$m_{train} = 10$$
* The data is generated from the linear function. 


```python
inputs, results, errors = run_experiment()
plot_results(inputs, results)
print("Training Error\tTest Error")
for k in errors.keys():
    print("{0[0]:<16.2f}{0[1]:.2f}".format(errors[k]))

Training Error	Test Error
0.00            0.00
0.00            0.00
0.00            0.00
```
<center>
<figure>
{% asset ml-generalization-exp-01.svg %}
</figure>
</center>


This is the easiest situation. The simplest model easily captures the main trend in the data. Since, we are already at the lowest possible test error with the linear model, therefore, it isn't necessary to increase the model complexity. Even if we do so, nothing really changes. 

Let's repeat the same experiment but this time let's generate the underlying data from the non-linear function (which is a 10th degree polynomial). 

Now we see that both training and test errors decrease as we increase model complexity. As soon as we fit a 10th degree polynomial, we are all set, no generalization error. In other words, in the noise-less case, our problem simply reduces to making sure that our model belongs to the same class as the data generating function.

### Experiment 2
* No noise in the data, $$\sigma = 0$$
* Small training dataset, $$m_{train} = 10$$
* Data generated from a 10th order polynomial (a non-linear process)


```python
inputs, results, errors = run_experiment(linear=False)
plot_results(inputs, results, ylim=(-4.0, 20))
print("Training Error\tTest Error")
for k in errors.keys():
    print("{0[0]:<16.2f}{0[1]:.2f}".format(errors[k]))

Training Error	Test Error
2.77            15430.96
0.64            12690.27
0.00            0.00

```
<center>
<figure>
{% asset ml-generalization-exp-02.svg %}
</figure>
</center>


In this case, we still have no noise in the measurements but we begin with a model that is outside of the class that generates the model and increase complexity eventually matching up with the model that generated the data. As expexted, since the patterns in the data are real, the simpler models tend to ignore them and underfit. Underfitting is characterized by poor performance on both training and test sets, as seen in the table. As we increase the model complexity, we can reduce errors on both training and test sets. Just to reiterate, the reason why training and test set performances improve concomitantly with complex models and in the absence of noise is that any patterns in the training data can be assumed to be genuine and hence a complex model can account for such patterns. 

Let's move into the real world now and introduce some noise.

### Experiment 3
* Noise in the data, $$\sigma = 1.0$$
* Relatively large training dataset, $$m_{train} = 100$$
* Data generated from a linear model


```python
inputs, results, errors = run_experiment(n_train=100, n_test=20, noise=1.0)
plot_results(inputs, results, ylim=(-2.0, 10.0), yticks=6)
print("Training Error\tTest Error")
for k in errors.keys():
    print("{0[0]:<16.2f}{0[1]:.2f}".format(errors[k]))

Training Error	Test Error
0.89            1.26
0.86            4.40
0.74            82477701.06
```
<center>
<figure>
{% asset ml-generalization-exp-03.svg %}
</figure>
</center>

We are in serious trouble right away. The first thing to notice is the widening gap between training and test errors as we increase moel complexity. The model with the lowest training error has the worst test error. The plot on the right shows clearly where these errors comes from i.e., test cases outside the range of training data. The training error decreases as we build more complex models because the model fit to the noisy patterns in the data. But the same model is completely off as soon as we step outside the training data range. Strikingly, the linear model, which is the simplest of the three, generalizes quite well. But we should also note that the underlying model that generated the data also belongs to the same class. In other words, the choice of correct model class with limited training data leads to much better generalization. 

Unfortunately, we are never truly aware of the mathematical model that generates the data. Let's see that in the  next experiment a bit more concretely.

### Experiment 4
* Noise in the data, $$\sigma = 1.0$$
* Relatively large training dataset, $$m_{train} = 100$$
* Data generated from a non-linear model


```python
inputs, results, errors = run_experiment(n_train=100, n_test=20, noise=1.0, linear=False)
plot_results(inputs, results, ylim=(-4.0, 20.0), yticks=6)
print("Training\tTest")
for k in errors.keys():
    print("{0[0]:<16.2f}{0[1]:.2f}".format(errors[k]))

Training	Test
1.92            10315.88
1.28            8977.04
0.92            213488.34
```
<center>
<figure>
{% asset ml-generalization-exp-04.svg %}
</figure>
</center>



The data comes from a non-linear model and still has noise in it, so we have a mix of real and unreal trends in the data. We want the model to learn the major trends but ignore the noisy fluctuations. 

We begin with a model outside of this class, i.e., a linear model. Unfortunately, it misses out on a key trend at the higher values of $$x$$, therefore, it performs quite badly on the test set. We increase the model complexity a bit and notice a simultaneous improvement on training and test set performance. Encouraged by this, we fit a more complex model. The training error goes down, but suprisingly the test error goes up significantly. We starting to see something interesting here, a possible sweet spot where trainig error isn't the minimum but the test error is! This will eventually lead us to the notion of bias variance trade-off, which we haven't discussed in detail in thise notebook. For now let's see what happens if we increase the size of training data. The question here is that can we gain test set performance by using complex model on a large dataset. 

### Experiment 5
* Noise in the data, $$\sigma = 1$$
* Relatively large training dataset, $$m_{train} = 10000$$
* Data generated from a non-linear model


```python
inputs, results, errors = run_experiment(n_train=10000, n_test=100, noise=1.0, linear=False)
plot_results(inputs, results, ylim=(-4.0, 10.0), yticks=6)
print("Training Error\tTest Error")
for k in errors.keys():
    print("{0[0]:<16.2f}{0[1]:.2f}".format(errors[k]))

Training Error	Test Error
1.99            7051.25
1.36            5910.58
1.02            979.75
```
<center>
<figure>
{% asset ml-generalization-exp-05.svg %}
</figure>
</center>

It seems that this strategy may be working. Both the training and test error goes down with increasing model complexity. So increasing the size of the training data is certainly useful here because generalization error is lowest for the most complex model. 

# Concluding Remarks
The experiments above indicate that with the limited nosiy training data, we can easily overfit and have a poor predictive performance on the new data. A simple model may give better performance in such cases. It would be less sensitive to the training data used. It will certainly not capture the true relationship (i.e., it will have high bias) but at the same time, a more complex model would be highly sensitive to the particular realization of the training dataset (see this in action by simply repeating experiment 3 without even changing anything). In other words, the model will have high variance. 

