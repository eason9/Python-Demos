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
#p = [random.uniform(-2,2) for i in range(draws)]
p = np.linspace(-2, 2, draws)
t = [g(p) for p in p]
alpha = .1
S = 2 #Number of neurons
m = 1 #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Stochastic Gradient approach with 10 Neurons use the settings:
'''
draws = 100
epochs = 2000
samples = 3
#p = [random.uniform(-2,2) for i in range(draws)]
p = np.linspace(-2, 2, draws)
t = [g(p) for p in p]
alpha = .1
S = 10 #Number of neurons
m = 1 #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Batch approach with 2 Neurons use the settings (takes awhile):
'''
draws = 100
epochs = 30000
samples = 1
#p = [random.uniform(-2,2) for i in range(draws)]
p = np.linspace(-2, 2, draws)
t = [g(p) for p in p]
alpha = .1
S = 2 #Number of neurons
m = draws #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

# For the Batch approach with 10 Neurons use the settings (takes awhile):
'''
draws = 100
epochs = 20000
samples = 1
#p = [random.uniform(-2,2) for i in range(draws)]
p = np.linspace(-2, 2, draws)
t = [g(p) for p in p]
alpha = .2
S = 10 #Number of neurons
m = draws #Size of Batches ("1"-stochastic Grad; "draws"-Batch)
'''

draws = 100
epochs = 2000
samples = 1
#p = [random.uniform(-2,2) for i in range(draws)]
p = np.linspace(-2, 2, draws)
t = [g(p) for p in p]
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
    epochs_MSE = []
    
    for epoch in range(epochs):
        
        # Generating Gradient Stores:
        G_s2a1 = []
        G_s2 = []
        G_s1p = []
        G_s1 = []

        for i in range(draws):
            
            # Forward Propagation:
            n_1 = W_1*p[i]+b_1
            a_1 = logsig(n_1)
            n_2 = np.dot(W_2, a_1)+b_2
            a_2 = purelin(n_2)
            
            # Backward Propagation:
            e = t[i] - a_2
            s_2 = -2*e
            f_prime = np.identity(S)*((1-a_1)*(a_1))
            s_1 = np.dot(f_prime, W_2.T*s_2)
            
            # Storing Gradients:
            G_s2a1.append(s_2*a_1.T)
            G_s2.append(s_2)
            G_s1p.append(s_1*p[i].T)
            G_s1.append(s_1)

            # Update Weights:
            if (i % m == 0) or (i == (len(p)-1)):
                W_2 = W_2 - (alpha*np.mean(G_s2a1, axis=0))
                b_2 = b_2 - (alpha*np.mean(G_s2, axis=0))
                W_1 = W_1 - (alpha*np.mean(G_s1p, axis=0))
                b_1 = b_1 - (alpha*np.mean(G_s1, axis=0))
                
                # Resetting Gradient Stores:
                G_s2a1 = []
                G_s2 = []
                G_s1p = []
                G_s1 = []

        # Calculating MSE for epoch:
        epoch_output = []
        for i in range(draws):
            epoch_output.append(NN_1S1(p[i], W_1, b_1, W_2, b_2))
        epoch_output = np.array([epoch_output]).T
        epoch_error = np.array([t]).T - epoch_output
        epoch_MSE = float(sum(epoch_error**2)/len(epoch_error))
        epochs_MSE.append(epoch_MSE)
        
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
        
    #Calculating Sample MSE:
    sample_output = []
    for i in range(draws):
        sample_output.append(NN_1S1(
                p[i], 
                best_epoch_W_1, 
                best_epoch_b_1, 
                best_epoch_W_2, 
                best_epoch_b_2
            ))
    sample_output = np.array([sample_output]).T
    sample_error = np.array([t]).T - sample_output
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
