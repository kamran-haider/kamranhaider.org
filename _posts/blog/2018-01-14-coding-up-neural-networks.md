---
layout : post
title: "Coding up a Toy Deep Neural Network"
excerpt: "Learning by doing is the best. I recently attempted implementing a [bare-bones deep neural network](https://github.com/kamran-haider/bbbp_ml_study/tree/master/code/toyNN) from scratch and it was so much fun."
category : posts
last_modified_at: 
tags: 
  - Python
  - Deep Learning
published: true
date: 2018-02-15
---

My interest in deep learning began after having conversations with [Bharath Ramsundar](http://rbharath.github.io/) 
(lead developer of [DeepChem](https://deepchem.io/)) and [Steven Kearnes](https://research.google.com/pubs/StevenKearnes.html) 
(a Google researcher working on applications of deep learning in drug discovery), last summer at 2017 Gordon Computer-aided 
Drug Design Conference. I was impressed by their work and by the fact that deep learning neural networks are remarkably 
successful in computing complex functions. In my own work, the ability to approximate complex functions is a routine part of the job, 
therefore, I was naturally interested. 

Within minutes of googling deep learning, I noticed how amazingly easy it is to get started in this area. If you have 
basic programming and maths skills, the technical barrier is quite low. There are great libraries and tools, such as 
[`Keras`](https://keras.io/), [`TensorFlow`](https://www.tensorflow.org/), [`Theano`](http://deeplearning.net/software/theano/), 
[`PyTorch`](http://pytorch.org/), and many others that get you started in building and training models very quickly. 
There is also an amazing amount of learning material available online in the form of [courses](https://www.deeplearning.ai/), 
[books](http://neuralnetworksanddeeplearning.com/index.html), and [blogs](http://colah.github.io/). 
I chose Andrew Ng's [Coursera deep learning specialization](https://www.deeplearning.ai/) and Michael Nielsen's online 
[book](http://neuralnetworksanddeeplearning.com/index.html) as starting points. 

To gain a better understanding of deep learning, especially the algorithmic aspects, I decided to spend 
some time coding up a simple [implementation](https://github.com/kamran-haider/toynn) of 
neural networks (named `toynn`) based on whatever I have learnt so far. Obviously, there are infinitely better 
implementations available in the tools that I mentioned above. However, the motivation behind creating `toynn` was 
to get a better understanding of how basic deep neural networks really work.

As I was coding this up, I was thinking about the idea of learning by doing. I would digress a little bit just to share an anecdote. 
I was visiting a collaborator at University of Cambridge in December 2012 and got lucky to catch a talk by the great 
[David Baker](https://www.bakerlab.org/), one of my most favorite scientists and one of the top researchers in the field of
protein design. He said, while talking about the motivation  behind designing proteins in the laboratory, 
**"We don’t know much about proteins, so we thought we should just create them 
to get a better understanding.”** His nonchalance felt even more impressive after he gave a fascinating talk on his recent work. 
I thought this must be an empowering feeling, i.e., create things you want to understand. This also reminds me of what Demis
Hassabis of [DeepMind](https://deepmind.com/) said in one of his talks, **"The ultimate expression of understanding something is to be able to recreate it."**, 
which he points out comes from Richard Feynman's words that **"What I cannot build, I do not understand"**.
Taking inspiration from this, I have tried to use this idea as a general principle to understand 
concepts. Obviously, it applies only in cases where there is a tractable way of creating something that directly or indirectly
sheds ligt on the concept I am trying to learn. That’s why despite the abundance of great deep learning libraries, I wanted to 
code up a neural network from scratch.

I wouldn't go ahead and give an introduction to deep learning or neural networks here and rather point readers to the 
[first chapter](http://neuralnetworksanddeeplearning.com/chap1.html) of Michael Nielsen's book. Here, I will just 
add a couple  of notes about my implementation. 

I am a huge fan of `scikit-learn` estimator API, which closely follows how machine learning projects are structured in general. 
Once the data has been adequately pre-processed, a machine learning task can be done with the following steps in `scikit-learn`:

* Choosing a model
* Fitting model to the data with `fit()` 
* Applying the trained model to new data with `predict()`
* Refine and fine-tune the model

Following this API, I created a module called `models` that consists of different types of neural networks that 
are supported. Currently, only one type is implemented which is called, `BasicDeepModel`. 
As you'd have guessed, each type of model has `fit()` and `predict()` functions. The `BasicDeepModel` itself is built from 
layers, whcih are implemented in a separate module `layers`. Currently, the distinction between layers is based on the non-linearities that 
are used to calculate activations of the constituent nodes. For example, two layers are supported; `Sigmoid` and 
`ReLU`, which use sigmoid and ReLU activation functions, respectively. To maintain consistency in the design, an `Input` 
layer is also implemented whose activations are initialized from the neural network inputs. The forward and backward methods 
for input layer don't really do anything. However, coding the inputs as a layers class allows me to write compact 
forward and backward propagation methods for the network. Finally, the `utils` module contains some useful functions,
such as loss functions, their derivatives, and network parameter initialization schemes. 

Now, let's see all of this in action. An example workflow for a binary classification problem would look like:

```python
from toynn.models import BasicDeepModel
from toynn.layers import *
from toynn.utils import *

training_data = "../toynn/tests/test_datasets/train_catvnoncat.h5"
test_data = "../toynn/tests/test_datasets/test_catvnoncat.h5"
train_x_orig, train_y, test_x_orig, test_y, classes = load_test_data(training_data, test_data)
num_px = train_x_orig.shape[1]

# Pre-processing of data
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

input_layer_nodes = num_px * num_px * 3
layers = [Input(input_layer_nodes), ReLU(20), ReLU(7), ReLU(5), Sigmoid(1)]
model = BasicDeepModel(train_x, train_y, layers, weight_initialization="custom")
model.fit(learning_rate=0.0075, n_epochs=2500)
predictions = model.predict(test_x)
```

`predictions` is an array consisting of the probability of belonging to the class for each data point. 
One can easily check the accuracy by converting these probabilities to class labels and then comparing with labels in 
`test_y`. See full example in a jupyter notebook [here](https://github.com/kamran-haider/toyNN/blob/master/examples/01-recognizing-cat-images.ipynb).
My starting point for this implementation was the material from Week 4 of the first course in Coursera deep learning specialization. 
I have checked the implementation using a dataset of cat images and reproduced the test accuracy of 0.8, which is identical 
to the implementation provided in the course. I also drew inspiration from another great and more comprehensive implementation
I found [here](https://github.com/cstorm125/sophia/blob/master/from_scratch.ipynb). Isn't it amazing that we are living 
in a world where people do cool stuff on Jupyter notebooks and then make it accessible to everyone?


The most fun part of coding up a neural network from scratch was to see backpropagation unravel as a set of matrix multiplications.
There is definitely a lot of room for improvement and enhancements in `toynn`, such as better documentation, unit tests and 
features such as regularization and advanced architectures. For now, I would continue diving a bit deeper into how neural 
networks can be tuned to solve various problems using some of the existing amazing tools, such as 
`PyTorch`, `Keras` and `TensorFlow`. Nevertheless, I am glad that I have `toynn` to play with whenever 
I needed to understand something through coding. I was also asked by at least one colleague that they 
would like to take a look at it and may be tear it apart and rebuild it to learn how it works. 
So I thought it would be a good idea to share it with everyone. 
If you are reading this and feel intrigued, feel free to take a look at the 
[code](https://github.com/kamran-haider/toynn), provide feedback or use it for your own practice.