# GOSR
The code and dataset for our paper: Graph Collaborative Optimization based Sequence Recommendation (https://ieeexplore.ieee.org/abstract/document/9714053). We have implemented our methods in Pytorch.

## Dependencies

- Python 3.8
- torch 1.9.0

## Usage 

### Datasets

You need to download the datasets required for the model via the following links:

Games: http://jmcauley .ucsd.edu/data/amazon

ML-1M: https://grouplens.org/datasets/movielens/1m/

LastFM: https://grouplens.org/datasets/hetrec-2011/


####Generate data

You need to run the file ```new_data.py``` to generate the data format needed for our model. The detailed commands 
can be found in ```load_{dataset}.sh```

You need to run the file ```generate_neg.py``` to generate data to speed up the test. You can set the 
data set in the file.

### Training and Testing 

Then you can run the file ```main-{dataset}.py``` to train and test our model. 
The detailed commands can be found in ```{dataset}.sh```
