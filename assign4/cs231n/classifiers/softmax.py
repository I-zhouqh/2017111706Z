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
    N=X.shape[0]
    C=W.shape[1]

    scores=X.dot(W)  # N*C
    
    for i in range(N):
        score_row=np.exp(scores[i,:])
        loss+= -np.log(score_row[y[i]]/np.sum(score_row))
        
        
        coef=1/np.sum(np.exp(scores[i,:]))
        for j in range(C):
            dW[:,j]+=coef * np.exp(scores[i,j])*X[i,:]
        dW[:,y[i]]-=X[i,:]
    
    loss=loss/N
        
    loss+=reg*np.sum(W*W)
    dW=dW/N
    dW+=2*reg*W
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

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

    
    N=X.shape[0]
    C=W.shape[1]
    exp_scores=np.exp(X.dot(W))  # N*C
    
    p=exp_scores[range(N),y]/np.sum(exp_scores,axis=1)
    loss+=np.sum(-np.log(p))/N
    loss+=np.sum(W*W)
    
    
    map_matrix=exp_scores/np.sum(exp_scores,axis=1).reshape(N,1)
    map_matrix[range(N),y]-=1
    dW=np.dot(X.T,map_matrix)
    dW/=N
    dW+=2*reg*W
    
    #coef=np.sum(np.exp(scores),axis=1)  # N*1
    
      # N*C    N*P             W:P*C
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
