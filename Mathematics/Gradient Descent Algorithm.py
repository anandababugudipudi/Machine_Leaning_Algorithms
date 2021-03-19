# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:43:47 2021

@author: anand
"""
# The algorithm starts at 3
cur_x =5
# Learning Rate
rate = 0.1
# This tells us when to stop the algorithm
precision = 0.5

previous_step_size = 1
# Maximum number of iterations
max_iters = 1000000
# iteration counter
iters = 0

# Gradient of our function
df = lambda x: 2*(x+5) 

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x
    # Gradient Descent
    cur_x = cur_x - rate * df(prev_x)
    previous_step_size = abs (cur_x - prev_x)
    iters = iters + 1
    print(f"Iteration{iters} \nX value is {cur_x}")
    
print(f"The local minimum occurs at {cur_x}")

# scipy.optimize.minimize() can be used to calculate complex functions





