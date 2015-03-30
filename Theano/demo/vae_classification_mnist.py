"""
Authors: 
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com>

#License: MIT
"""

"""This script load an auto-encoder on the MNIST dataset and train an ?LP with Rectifiers"""

#python trainmnist.py -s mnist.npy
import theano
import theano.tensor as T

import numpy as np
import argparse
import time
import gzip, cPickle

import VariationalAutoencoder
from learning_rule import RMSPropMomentum

from blocks.bricks import MLP, Rectifier, Softmax, Identity, Tanh
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.roles import WEIGHT, BIAS
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter

def get_params(cost):
    cg=ComputationGraph(cost)
    return VariableFilter(roles=[WEIGHT])(cg.variables)+VariableFilter(roles=[BIAS])(cg.variables)

def sauvegarde(cost,path):
    cg=ComputationGraph([cost])
    liste_weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    liste_bias= VariableFilter(roles=[BIAS])(cg.variables)
    weights=[liste_weights[i].get_value().astype(np.float32) for i in xrange(len(liste_weights))]
    biases =[liste_bias[i].get_value().astype(np.float32) for i in xrange(len(liste_bias))]
    save_params={'weights':weights, 'biases':biases}
    fichier=open(path,'wb')
    cPickle.dump(save_params, fichier)
    fichier.close()

def build_classifier(dimension):
    activations = [ Rectifier(), Rectifier(), Softmax()]
    mlp = MLP(activations, [dimension, 800, 800, 10],
          weights_init=IsotropicGaussian(0.01),
          biases_init=Constant(0))
          
    mlp.initialize()
    return mlp

def testing(n_epochs=500, batch_size=10, learning_rate=0.01, momentum=0.2):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

    args = parser.parse_args()

    print "Loading MNIST data"
    #Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

    f = gzip.open('/data/lisa/data/mnist/mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
    f.close()
    n_train_batches = len(x_train)/batch_size
    n_valid_batches = len(x_valid)/batch_size

    data = x_train
    dimZ = 20
    HU_decoder = 400
    HU_encoder = HU_decoder

    batch_size = batch_size
    L = 1
    learning_rate = 0.01
    if args.double:
        print 'computing hidden layer to train new AE on'
        prev_params = np.load(args.double)
        data = (np.tanh(data.dot(prev_params[0].T) + prev_params[5].T) + 1) /2
        x_test = (np.tanh(x_test.dot(prev_params[0].T) + prev_params[5].T) +1) /2

    [N,dimX] = data.shape
    encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)

    encoder.createGradientFunctions()
    print "Initializing weights and biases"
    encoder.initParams()
    x=T.matrix('features')
    y = T.ivector('targets')
    # define MLP
    mlp = build_classifier(20)
    prediction = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(), prediction)
    error_rate = MisclassificationRate().apply(y.flatten(), prediction)
    validate_model=theano.function([x, y], error_rate,
				 on_unused_input='warn',
                 allow_input_downcast=True)

    #training with momentum
    """
    rmsprop = RMSPropMomentum(learning_rate=1e-4, momentum=0.)
    params= get_params(cost)
    grads = T.grad(cost , params)
    train_model = theano.function([x, y], cost, updates=rmsprop.get_updates(learning_rate, params, grads),
                  on_unused_input='warn',
                  allow_input_downcast=True)
    """
    params = get_params(cost)  
    velocity = [ theano.shared(np.cast[theano.config.floatX](0.*params[i].get_value())) for i in xrange(len(params))]
    grads = T.grad(cost , params)
    updates=[]
    for param_i, grad_i in zip(params, grads):
	    updates.append((param_i, param_i  -learning_rate*grad_i))
    """
    for velocity_i, grad_i in zip(velocity, grads):
	    updates.append((velocity_i, momentum*velocity_i - learning_rate*grad_i))
    """
    train_model = theano.function([x,y],
                                  cost, updates=updates,
                                  allow_input_downcast=True
                                  )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    #file to look at the evolution overfitting/underfitting
    

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    #load
    fichier = open("demo/sauvegarde_temp.txt", 'rb')
    params = cPickle.load(fichier)
    encoder.params = params
    fichier.close()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # compute the output
        output_train = np.asarray(encoder.encode(data))
        shape_1 = output_train.shape
        output_train = output_train.reshape((shape_1[0], shape_1[3], shape_1[2]))
        output_valid = np.asarray(encoder.encode(x_valid))
        shape_1 = output_valid.shape
        output_valid = output_valid.reshape((shape_1[0], shape_1[3], shape_1[2]))
        for index in xrange(0,n_train_batches-1):
            minibatch_avg_cost = train_model(output_train[index], t_train[index*batch_size:(index+1)*batch_size])
            validation_losses = []
            for i in xrange(n_valid_batches-1):
                validation_losses.append( 
                                validate_model(output_valid[index], t_valid[index*batch_size:(index+1)*batch_size]))
            this_validation_loss = np.mean(validation_losses)
            print "Train :"+str(minibatch_avg_cost)
            print "Valid :"+str(this_validation_loss*100)+"%"

            if this_validation_loss < best_validation_loss :
                #save params
                sauvegarde(cost,'demo/semi_supervised_classification')
 
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
if __name__ == '__main__':
    testing()


