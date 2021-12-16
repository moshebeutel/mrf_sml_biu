import numpy as np
from numpy import random
from numpy.core.fromnumeric import shape
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime

grid_shape = (5,5)
grid_size = grid_shape[0]*grid_shape[1]

def get_row_col(i):
    assert i >= 0
    assert i < grid_size
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

def phi_x_x(x:np.array, i:int, j:int)->int:
    irow, icol = get_row_col(i)
    jrow, jcol = get_row_col(j)
    return int(x[irow, icol] == x[jrow, jcol])

def phi_x_y(x:np.array, y:np.array ,i:int):
    irow, icol = get_row_col(i)
    return -0.5*(x[irow,icol] - y[irow, icol])**2.0


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

def init_x_using_gibbs_sampling(x:np.array, T:int, val:int):
    x = np.random.randn(grid_size)
    samples = np.zeros(grid_size,T)
    for t in range(T):
        for i in range(grid_size):
            samples[get_row_col(i),t] = sample_from_p_val_given_blanket(x,val,i)
            # x is updated at every iteration so
            #     x_i_t is sampled given all x_j<i_t and all x_j>i_t-1  
            x[get_row_col(i)] = val if np.mean(samples[i,0:t+1]) > 0.5 else 1-val
    return x



def estimate_using_gibbs_sampling(x:np.array, T:int, exact_marginals):
    samples = np.zeros_like(grid_size,T)
    p_hat = np.zeros_like(grid_size)
    error = np.zeros_like(grid_size,T)
    for t in range(T):
        for i in range(grid_size):
            samples[get_row_col(i),t] = sample_from_p_val_given_blanket(x,1,i)
            p_hat[i]=  np.mean(samples[i,0:t+1]) 
        error[:,t] = np.sum(np.square(p_hat - exact_marginals),axis=0)
    #TODO plot
    plot_error(error, "Gibbs Eampling Estimation Error")
    
def plot_error(error, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(error)
    plt.savefig(f'{title} {datetime.utcnow()}')
    plt.show()

def pseudo_p_x_given_y(x:np.array, y:np.array, E:list):
    sum_phi_x_y = 0
    for i in range(grid_size):
        sum_phi_x_y += phi_x_y(x,y,i)
    sum_phi_x_x = 0
    for k,l in E:
        sum_phi_x_x += phi_x_x(x,k,l)
    
    return np.exp(sum_phi_x_x + sum_phi_x_y)


def compute_exact_marginals(y:np.array, E:list, val:int, index:int):
    x_options = [list(t) for t in (product(*([0,1],)*grid_size))]
    x_val_options = [l for l in x_options if l[index]==val]
    x_other_val_options = [l for l in x_options if l[index]==1-val]
    sum_val = 0
    sum_other_val = 0
    for x in x_val_options:
        sum_val += pseudo_p_x_given_y(x,y,E)
    for x in x_other_val_options:
        sum_other_val += pseudo_p_x_given_y(x,y,E)
    return sum_val / (sum_val + sum_other_val)

def estimate_using_mean_field_approximation(x:np.array, y:np.array, T:int, exact_marginals):
    q = np.zeros_like(grid_size)
    error = np.zeros_like(grid_size,T)
    for t in range(T):
        for i in range(grid_size):
            markov_blanket = get_markov_blanket(i)
            q[i]=  np.exp(phi_x_y(x,y,i)) +  np.sum([q[j]*phi_x_x(i,j) for j in markov_blanket])
        error[:,t] = np.sum(np.square(q - exact_marginals),axis=0)
    plot_error(error, "Mean Field Approximation Error")


def normal_dequantization(x:np.array):
    y = np.random.normal(loc=x)
    return y

def create_edges(x:np.array):
    edges = []
    index = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if j < grid_shape[1] -1:
                edges.append((index,index+1))
            if i < grid_shape[0] - 1:
                edges.append((index,index + grid_shape[1]))
            index+=1
    return edges

    

x = np.random.randint(low=0, high=2,size=grid_shape)
E = create_edges(x)

print(x)
print('***')
print(len(E))
print('***')
print(E)
print('***')
print([phi_x_x(x, *e) for e in E])




