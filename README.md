# Profiling-and-Hyperparameter-tuning

This notebook will use two practical tools to facilitate deep learning analysis process.

* The first step is to use the [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler) to analyze an input pipeline for a slow running program, then add a few lines of code to improve performance. The profiler is built-in to [TensorBoard](https://www.tensorflow.org/tensorboard), a popular visualization tool (parts of this work with both TensorFlow and PyTorch).

* Next, we will use [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) to tune the hyperparameters for a text classifier. Keras Tuner is an open-source hyperparamter tuning package that works with TensorFlow, scikit-learn, and other frameworks.

The notebook is intially run on the [google colab](https://colab.research.google.com). The data is also downloaded from the internet. To run the whole notebook, simply uploading it to [google colab](https://colab.research.google.com) will make everything go fine.

## Dataset

The [dataset](https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz) contains programming questions extracted from Stack Overflow. Each question ("How do I sort a dictionary by value?") is labeled with exactly one tag (`Python`, `CSharp`, `JavaScript`, or `Java`).

In data preprocessing step, we use [text_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory). This utility creates a labeled `tf.data.Dataset` from a directory structure as follows. 

The training data is splited to get 20% as validation data. Data is fed in model in batches of size 32.

Next, we standardize, tokenize, and vectorize the data using the preprocessing.TextVectorization layer.

Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset.

Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual words by splitting on whitespace).

Vectorization refers to converting tokens into numbers so they can be fed into a neural network. There are two ways of vectorization text data: int mode and binary mode. Int mode converts each token to integer indices, while binary mode outputs one-hot indicies corresponding to distinct tokens by having 1s on diffirent dimensions.


## Part 1: Profile a slow text classifier using the TensorFlow Profiler

The most common peformance bottleneck in a DL program is the input pipeline. In a nutshell, modern GPUs are so fast they often sit idle while waiting for data to be loaded off disk, and/or preprocessed by the CPU (informally, this is called "GPU starvation").

We use the [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler) to identify a performance problem, and [tf.data](https://www.tensorflow.org/guide/data_performance) to fix it. 

Before optimizing the data input pipeline, the tensorboard interface showed that 82.2% of the time spent on the device is waiting for the input. Thus we need to configure the dataset to optimize the input pipeline by cache as prefetch method.

![alt text](https://user-images.githubusercontent.com/79208856/112758314-41123e80-9020-11eb-9d2e-e552d2c769e7.png)

After optimizing input pipeline, the data input time is much shorter.

![alt text](https://user-images.githubusercontent.com/79208856/112758857-fb0aaa00-9022-11eb-97fd-bad271f612df.png)

By prefetching data to the memory, we reduce the average device step time from 12.1ms to 2.6ms.

## Part 2: Hyperparameter tuning

This part uses keras tuner to improve model. We applied several strategies as below:

* Explore alternative sizes for the Embedding and Conv1D layers (perhaps more or fewer neurons are helpful?) 
* Explore different arrangements and types of layers.
* Experiment with RNNs.

After tuning the model, we get validation accuracy of 82%.




