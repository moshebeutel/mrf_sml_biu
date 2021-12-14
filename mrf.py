import numpy as np
from numpy import random

grid_shape = (5,5)

def get_row_col(i):
    assert i >= 0
    assert i < grid_shape[0]*grid_shape[1]
    irow,icol = i // grid_shape[1], i % grid_shape[1]
    assert icol < grid_shape[1]
    assert irow < grid_shape[0]
    return irow, icol

def get_index_using_row_col(row,col):
    assert 0 <= row < grid_shape[0]
    assert 0 <= col < grid_shape[1]
    return row * grid_shape[1] + col

def get_markov_blanket(i):
    row, col = get_row_col(i)
    return [get_index_using_row_col(k,l) for k in [row-1,row,row+1] for l in [col -1, col, col + 1] \
         if 0<=k<grid_shape[0] and 0<=l<grid_shape[1]]

def phi(x:np.array, i:int, j:int):
    irow, icol = get_row_col(i)
    jrow, jcol = get_row_col(j)
    return int(x[irow, icol] == x[jrow, jcol])

def phi_by_val(x:np.array, val:int, j:int):
    assert val in [0,1]
    jrow, jcol = get_row_col(j)
    return int(val == x[jrow, jcol])

def p_val_given_blanket(x:np.array, val:int, index:int):
    blanket = get_markov_blanket(index)
    sum_val = 0
    for j in blanket:
        sum_val += phi_by_val(x,val,j)
    other_val = 1-val
    sum_other_val=0
    for j in blanket:
        sum_other_val += phi_by_val(x,other_val,j)
    return sum_val / (sum_val + sum_other_val)

def sample_from_p_val_given_blanket(x:np.array, val:int, index:int):
    prob = p_val_given_blanket(x,val,index)
    r = random.rand()
    return val if prob < r else 1-val

def gibbs_sampling(x:np.array, T:int):
    samples = np.zeros((grid_shape[0]*grid_shape[1],T))
    for t in range(T):
        for i in range(grid_shape(0)*grid_shape(1)):
            samples[get_row_col(i),t] = sample_from_p_val_given_blanket(x,1,i)
            # x is updated at every iteration so
            #     x_i_t is sampled given all x_j<i_t and all x_j>i_t-1  
            x[get_row_col(i)] = 1. if np.mean(samples[i,0:t+1]) > 0.5 else 0.



def create_edges(x:np.array):
    return [(x[get_row_col(i)],x[get_row_col(j)])  for i in range(grid_shape[0]) for j in range(grid_shape[1])]

x = np.random.randint(low=0, high=2,size=grid_shape)
E = create_edges(x)

print(x)
print('***')
print(len(E))
print('***')
print(E)
print('***')
print([phi(x, *e) for e in E])




