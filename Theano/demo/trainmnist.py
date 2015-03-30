"""
Authors: 
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com>

#License: MIT
"""

"""This script trains an auto-encoder on the MNIST dataset and keeps track of the lowerbound"""

#python trainmnist.py -s mnist.npy

import VariationalAutoencoder
import numpy as np
import argparse
import time
import gzip, cPickle

print 'kikou'

parser = argparse.ArgumentParser()
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

args = parser.parse_args()

print "Loading MNIST data"

f = gzip.open('/data/lisa/data/mnist/mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

dimZ = 20
HU_decoder = 400
HU_encoder = HU_decoder

batch_size = 100
L = 1
learning_rate = 0.01

if args.double:
    print 'computing hidden layer to train new AE on'
    prev_params = np.load(args.double)
    data = (np.tanh(data.dot(prev_params[0].T) + prev_params[5].T) + 1) /2
    x_test = (np.tanh(x_valid.dot(prev_params[0].T) + prev_params[5].T) +1) /2

[N,dimX] = data.shape
encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)



lower_bound = -np.inf
best_params = None

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
encoder.initParams()
lowerbound = np.array([])
testlowerbound = np.array([])
begin = time.time()

for j in xrange(1000):
    encoder.lowerbound = 0
    print 'Iteration:', j
    encoder.iterate(data)
    end = time.time()
    print("Iteration %d, lower bound = %.2f,"
          " time = %.2fs"
          % (j, encoder.lowerbound/N, end - begin))
    begin = end
    temp = encoder.getLowerBound(x_test)
    if temp > lower_bound:
        "kikou"
        lower_bound=temp
        best_params = np.copy(encoder.params)
        fichier = open("demo/sauvegarde.txt",'wb')
        cPickle.dump(encoder.params, fichier)
        fichier.close()
    print "Test :"+str(temp)
