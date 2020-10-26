# Neural architecture growing, pruning and search

This project is the result of my end-of-study internship in the Linkmedia Team in INRIA Rennes in 2020.

The thesis of this project will be accessible at intranet.insa-rennes.fr/informations-etudiants/memoires-de-master-et-de-pfe-en-ligne.html . 

## Source code

This project is based on the source code of yigitcankaya/Shallow-Deep-Networks from the ICML 2019 Paper "Shallow-Deep Networks: Understanding and Mitigating Network Overthinking"

It also uses some functions and code from mil-ad/snip that is a Pytorch implementation of the paper SNIP: Single-shot Network Pruning in Pytorch. 

## Dependencies

In order to run this project, the requirements are:
- Python 3.7
- PyTorch 1.0
- CUDA 8.0
- CUDNN 7.5
- Matplotlib Pyplot

## Datasets

The experiments made are based on the CIFAR10 dataset that is automatically downloaded from Pytorch base if it is not found in the current working directory.

## Run the code

The main file to run the experiments is the *iterative_experiments.py* file. The command line is the `python iterative_experiments.py`.

In this file are set the different hyperparameters of the experiments.

The hyperparameters are set in the **main** function:
- the type of network you want to run 
    - 'dense' to create a dense growing network with internal classifiers,
    - 'iterative' to create a feed-forward growing network with internal classifiers,
    - 'full_ic' to create a feed-forward network with internal classifiers,
    - 'full' to create a feed-forward network.

- The type of training you want to run
    - '0' to use the standard training method where the network is grown at set epochs,
    - '1' to use a yet to explore training using a loss based criterion for the growing.

- A tuple containing the pruning informations
    - a boolean coding the use or not of pruning,
    - an array of the size of the number of internal classifiers containing the ratio of kept connections after each pruning,
    - the size of the batch used for the pruning

- An array representing the structure of the network used:
    - The network will, at the end of the training, be composed of the same number of cells than the number of element in the array. A cell is here defined as two Conv-BatchNorm-ReLU layers with a connection skipping the second layer (similar to a ResNet cell).
    - If the element value is 0 then the cell does not include an Internal Classifier and it does if the value is 1.

Some hyperparameters are also defined in the **train_model** function.
- The number of epochs is defined by modifying the $params['epochs']$ parameter.
- The milestones for the learning rate scheduler are defined by modifying the $params['milestones']$ parameter by giving it an array containing the epochs where the learning rate is modified.
- The gammas for the learning rate scheduler represent the factor by which the current learning rate will be multiplied. They are set by modifying the $params['gammas']$ paramater by giving it an array of the same size of the milestones containing the factors.
- The learning rate is by default set to 0.01 and can be set by modifying the $params['learning_rate']$ paramater.

The following parameters are defined in the *train_params* $dict$ in the **train_model** function.
- The growth epochs are defined by an array containing the epochs where the growing will take place.
- The epochs where the pruning will be performed are defined by an array containing the said epochs.
- The type of pruning is defined by a number:
    - '0' for the skip_layer policy, pruning ony the last bloc. A bloc is here defined as the sequence of cells between two internal classifiers.
    - '1' for normal pruning, meaning pruning the network as a whole without taking in account the different blocs.
    - '2' for the iterative policy, pruning each bloc multiple time with a decreasing number of kept connections.
- A boolean coding the reinitialisation of the pruned part,
- An array of the same size than the number of blocs containing the minimum ratio that will not be exceeded by pruning those blocs. This parameter is relevant only for the '2'/iterative policy of pruning.


The previous explained setup, create, train from scratch and save a network while also printing some interesting informations. However, it is also possible to load an already trained network.

In the command shell, you can run `python iterative_experiments.py -l [names of the networks to load]`. This will load the called network and print the final informations about this network.
