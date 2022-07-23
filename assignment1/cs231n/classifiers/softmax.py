from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
      term = 0.0
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      for j in range(num_classes):
        term += np.exp(scores[j])
      for j in range(num_classes):
        dW[:,j] += np.exp(scores[j])/term * X[i].T
      loss += (np.log(term) - correct_class_score)
      dW[:,y[i]] -= X[i].T
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W*W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = scores[range(num_train),list(y)].reshape(-1,1)
    scoresexp = np.exp(scores)
    correctexp = np.exp(correct_class_score)
    sigma = np.sum(scoresexp,axis=1).reshape(-1,1)
    term2 = -np.log(correctexp/sigma)

    loss = np.sum(term2)/num_train + reg * np.sum(W*W)
    
    term3 = scoresexp/sigma
    term3[range(num_train),list(y)] -= 1

    dW = np.dot(X.T, term3)/num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
