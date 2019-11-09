# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:54:54 2019

@author: Charles Garrett Eason
ML2
HW6
"""
#%%Packages

import numpy as np
import matplotlib.pyplot as plt 

#%% Functions

# Logsigmoid:
def logsig(n):
    a = 1/(1+np.exp(-n))
    return(a)
    
# Pureline:
def purelin(n):
    a = n
    return(a)
    
#Positive Linear:
def poslin(n):
    if n < 0:
        a = 0
    elif n >= 0:
        a = n
    return(a)

# Function to be approximated:
def g(p):
    g = np.exp(-abs(p))*np.sin(np.pi*p)
    return(g)

# Execution of 1-S-1 Neural Network:
def NN_1S1(p, W_1, b_1, W_2, b_2):
    n_1 = W_1*p+b_1
    a_1 = logsig(n_1)
    n_2 = np.dot(W_2, a_1)+b_2
    a_2 = purelin(n_2)
    return(float(a_2))

#%% User Defined Variables

# For the Stochastic Gradient approach with 2 Neurons use the settings:
'''
draws = 100
epochs = 2000
samples = 3
#p = np.linspace(-2, 2, draws)
p = np.random.uniform(low=-2, high=2, size=draws)
t = g(p)
alpha = .1
S = 2 #Number of neurons
m = 1 #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Stochastic Gradient approach with 10 Neurons use the settings:
'''
draws = 100
epochs = 2000
samples = 3
#p = np.linspace(-2, 2, draws)
p = np.random.uniform(low=-2, high=2, size=draws)
t = g(p)
alpha = .1
S = 10 #Number of neurons
m = 1 #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Batch approach with 2 Neurons use the settings (takes awhile):
'''
draws = 100
epochs = 30000
samples = 1
#p = np.linspace(-2, 2, draws)
p = np.random.uniform(low=-2, high=2, size=draws)
t = g(p)
alpha = .1
S = 2 #Number of neurons
m = draws #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Batch approach with 10 Neurons use the settings (takes awhile):
'''
draws = 100
epochs = 20000
samples = 1
#p = np.linspace(-2, 2, draws)
p = np.random.uniform(low=-2, high=2, size=draws)
t = g(p)
alpha = .2
S = 10 #Number of neurons
m = draws #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

draws = 100
epochs = 2000
samples = 3
#p = np.linspace(-2, 2, draws)
p = np.random.uniform(low=-2, high=2, size=draws)
t = g(p)
alpha = .1
S = 10 #Number of neurons
m = 1 #Size of Batches ("1"-stochastic Grad; "draws"-Batch)

#%% 1-S-1 Function Approximation Neural Network

# 1-S-1 Network:
print('Running . . .')
for sample in range(samples):
    
    # Initalizing Random Weights between -.5 and .5
    W_1 = np.random.uniform(low=-.5, high=.5, size=(S,1))
    b_1 = np.random.uniform(low=-.5, high=.5, size=(S,1))
    W_2 = np.random.uniform(low=-.5, high=.5, size=(S,1)).T
    b_2 = np.random.uniform(low=-.5, high=.5, size=(1,1))
    
    # Epoch MSE Store:
    epochs_MSE = np.zeros(epochs)
    
    for epoch in range(epochs):
        
        # Generating Gradient Stores:
        G_s2a1 = np.zeros((m, S))
        G_s2 = np.zeros(m)
        G_s1p = np.zeros((m, S))
        G_s1 = np.zeros((m, S))
        b_count = 0

        for i in range(1, draws+1):
            
            # Forward Propagation:
            n_1 = W_1*p[i-1]+b_1
            a_1 = logsig(n_1)
            n_2 = np.dot(W_2, a_1)+b_2
            a_2 = purelin(n_2)
            
            # Backward Propagation:
            e = t[i-1] - a_2
            s_2 = -2*e
            f_prime = np.identity(S)*((1-a_1)*(a_1))
            s_1 = np.dot(f_prime, W_2.T*s_2)
            
            # Storing Gradients:
            G_s2a1[b_count] = (s_2*a_1).T
            G_s2[b_count] = s_2
            G_s1p[b_count] = (s_1*p[i-1]).T
            G_s1[b_count] = s_1.T
            b_count += 1

            # Update Weights in Batch:
            if (i % m == 0):
                W_2 = W_2 - (alpha*np.mean(G_s2a1, axis=0))
                b_2 = b_2 - (alpha*np.mean(G_s2, axis=0))
                W_1 = W_1 - (alpha*np.mean(G_s1p, axis=0)).reshape(S,1)
                b_1 = b_1 - (alpha*np.mean(G_s1, axis=0)).reshape(S,1)
                
                # Resetting Gradient Stores:
                G_s2a1 = np.zeros((m, S))
                G_s2 = np.zeros(m)
                G_s1p = np.zeros((m, S))
                G_s1 = np.zeros((m, S))
                b_count = 0
            
            # Updating Weights Using Remainder Gradients:
            elif (i == draws):
                W_2 = W_2 - (alpha*np.mean(G_s2a1[:i % m], axis=0))
                b_2 = b_2 - (alpha*np.mean(G_s2[:i % m], axis=0))
                W_1 = W_1 - (alpha*np.mean(G_s1p[:i % m], axis=0)).reshape(S,1)
                b_1 = b_1 - (alpha*np.mean(G_s1[:i % m], axis=0)).reshape(S,1)

        # Calculating MSE for epoch:
        epoch_output = np.zeros(draws)
        for i in range(draws):
            epoch_output[i] = NN_1S1(p[i], W_1, b_1, W_2, b_2)
        epoch_error = t - epoch_output
        epoch_MSE = float(sum(epoch_error**2)/len(epoch_error))
        epochs_MSE[epoch] = epoch_MSE
        
        # Selecting Best Epoch:
        if epoch == 0:
            best_epoch_MSE = epoch_MSE
            best_epoch_W_1 = W_1
            best_epoch_b_1 = b_1
            best_epoch_W_2 = W_2
            best_epoch_b_2 = b_2  
        elif epoch_MSE < best_epoch_MSE:
            best_epoch_MSE = epoch_MSE
            best_epoch_W_1 = W_1
            best_epoch_b_1 = b_1
            best_epoch_W_2 = W_2
            best_epoch_b_2 = b_2
        else:
            pass
        
        print('epoch: %s ; MSE: %s' % (epoch, best_epoch_MSE))
        
    #Calculating Sample MSE:
    sample_output = np.zeros(draws)
    for i in range(draws):
        sample_output[i] = NN_1S1(
                p[i], 
                best_epoch_W_1, 
                best_epoch_b_1, 
                best_epoch_W_2, 
                best_epoch_b_2
            )
    sample_error = t - sample_output
    sample_MSE = float(sum(sample_error**2)/len(sample_error))
    
    # Selecting Best Sample:
    print('sample: %s ; MSE: %s' % (sample, sample_MSE))
    if sample == 0:
        best_sample_MSE = sample_MSE
        best_W_1 = best_epoch_W_1
        best_b_1 = best_epoch_b_1
        best_W_2 = best_epoch_W_2
        best_b_2 = best_epoch_b_2
        best_output = sample_output    
    elif sample_MSE < best_sample_MSE:
        best_sample_MSE = sample_MSE
        best_W_1 = best_epoch_W_1
        best_b_1 = best_epoch_b_1
        best_W_2 = best_epoch_W_2
        best_b_2 = best_epoch_b_2
        best_output = sample_output
    else:
        pass
    if sample_MSE < .0001:
        print('Solution Found!')
        break
    
#%% Plots

# Function Output Plot:
plt.figure('NN output v. g(p)')
plt.plot(p, t, '.', label='g(p)') 
plt.xlabel('p') 
plt.ylabel('g') 
#plt.show()

# NN Approximation Plot:
plt.plot(p, best_output, '.', label='%s Neuron NN Output' % S) 
plt.xlabel('p') 
plt.ylabel('Output')
plt.legend(loc='upper left')
plt.show()

# MSE by Epoch Plot:
plt.figure('MSE by Epoch')
plt.loglog(list(range(epochs)), epochs_MSE) 
plt.xlabel('ln(Epochs)') 
plt.ylabel('ln(MSE)')
plt.show()

print('Execution Complete!')

#%% Out of Sample Plots
'''
x = np.random.uniform(low=-5, high=5, size=draws*10)
y = g(x)

yhat = np.zeros(draws*10)
for i in range(draws*10):
    yhat[i] = NN_1S1(
            x[i], 
            best_epoch_W_1, 
            best_epoch_b_1, 
            best_epoch_W_2, 
            best_epoch_b_2
        )
    
# Function Out of Sample Output Plot:
plt.figure('Out of Sample NN output v. g(x)')
plt.plot(x, y, '.', label='g(x)') 
plt.xlabel('x') 
plt.ylabel('y') 
#plt.show()
   
# NN Out of Sample Approximation Plot:
plt.plot(x, yhat, '.', label='%s Neuron NN Output' % S) 
plt.xlabel('x') 
plt.ylabel('yhat')
plt.legend(loc='upper left')
plt.show()
'''
