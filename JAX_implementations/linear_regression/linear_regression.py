import jax
import jax.numpy as jnp

def linear_regressor(params, inputs):
    '''
        Linear Regression Model
        params: (w, b) tuple of weights and biases
        inputs: input data 
    '''
    w, b = params
    return w * inputs + b

def loss_fn(params, inputs, targets):
    '''
        Mean Squared Error Loss Function 
    '''
    preds = linear_regressor(params, inputs)
    return jnp.mean((preds - targets) ** 2)

def update(params, inputs, targets, lr=0.1):
    return params - lr * jax.grad(loss_fn)(params, inputs, targets)