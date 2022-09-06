from os import sep
from scipy.stats import norm
from itertools import combinations
import numpy as np
import math
from causallearn.search.ConstraintBased.PC import pc


np.random.seed(7)
n_node,n_sample = 7,2000
digma =0.8
X1 = np.random.normal(loc=0.0, scale=digma, size=n_sample)
X2 = np.random.normal(loc=0.1, scale=digma+0.1, size=n_sample)
X3 = 0.5*X1+ 0.5*X2 + np.random.normal(loc=0.0, scale=digma, size=n_sample)
X4 = 0.5*X1+ 0.5*X2 + np.random.normal(loc=0.0, scale=digma, size=n_sample)
X5 = np.random.normal(loc=0.02, scale=digma+0.3, size=n_sample)
X6 = 0.5*X3+ 0.5*X4 + 0.5*X5 + np.random.chisquare(7)
X7 = 0.5*X3+ 0.5*X4 + np.random.normal(loc=0.0, scale=digma, size=n_sample)
data= np.vstack([X1,X2,X3,X4,X5,X6,X7])

var_num = data.shape[0]
alpha = 0.05
adj_mat = np.ones([var_num, var_num]) - np.eye(var_num)
cov_ = np.cov(data)

def fisher_Z_test(n_sample, alpha, cond_idx, var_idx, cov_):
        
        comb_var_idx = np.append(var_idx, cond_idx)

        if cond_idx.shape[0] == 0:
            cor_var = cov_[var_idx[0], var_idx[1]]
        else:
            precision_matrix = np.linalg.pinv(cov_[np.ix_(comb_var_idx, comb_var_idx)])
            cor_var = - precision_matrix[0, 1] / np.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
        cor_var = min(0.99999, max(cor_var, -0.99999))
        z_trans = 0.5 * math.log((1 + cor_var) / ( 1-cor_var))
        X = math.sqrt(n_sample - comb_var_idx.shape[0] - 3) * abs(z_trans) 
        p_value = 2 * (1 - norm.cdf(abs(X)))
        if p_value > alpha:
            return p_value, True
        else:
            return p_value, False

def skeleton_discovery(cov_, adj_mat, var_num, n_sample, alpha):
    depth = -1
    sep_set = []
    for i in range(var_num):
        sep_set.append([])
        for j in range(var_num):
            sep_set[i].append([])
    while max(np.sum(adj_mat != 0, axis = 1)) - 1 > depth:
        depth += 1
        for x in range(var_num):
            adj_vars, neigh_num = get_neigh(adj_mat, x)
            if neigh_num < depth - 1:
                continue
            for y in adj_vars:
                neigh_x_without_y = np.delete(adj_vars, np.where(adj_vars == y))

                for S in combinations(neigh_x_without_y, depth):
                    _, flag = fisher_Z_test(n_sample, alpha, np.array(S), np.array([x, y]), cov_)
                    if not flag:
                        continue
                    else:
                        sep_set[x][y] = sep_set[x][y] + list(S)
                        sep_set[y][x] = sep_set[y][x] + list(S)
                        adj_mat[x, y] = 0
                        adj_mat[y, x] = 0
    print(adj_mat)
    return adj_mat, sep_set

def get_neigh(adj_mat, var_idx):
    neigh_set = np.where(adj_mat[var_idx, :] != 0)
    return neigh_set[0], len(neigh_set[0])

def get_v_struct(adj_mat, sep_set, var_num):
    v_struct = []
    tri_nodes = []
    collidors = []
    for i in range(var_num):
        triu = np.triu(adj_mat, 1)
        a = np.where(triu[i, :] != 0)
        b = np.where(triu[:, i] != 0)
        if len(a[0]) + len(b[0]) >= 2:
            neigh_collidor = np.append(a[0], b[0])
            attackers = combinations(neigh_collidor, 2)
            for attacker in attackers:
                x = min(attacker)
                y = max(attacker)
                tri_nodes.append((x, i, y))
                if adj_mat[x, y] == 0:
                    if (i) not in sep_set[x][y]:
                        tri_nodes.pop()
                        collidors.append(i)
                        v_struct.append((x, i, y))
                        adj_mat[i, x] = -1
                        adj_mat[i, y] = -1
    return adj_mat, v_struct, list(set(collidors)), tri_nodes

def orient_edges(adj_mat, v_struct, collidors):
    
    for collidor in collidors:
        attackers = []
        for struct in v_struct:
            if struct[1] == collidor:
                attackers.append(struct[0])
                attackers.append(struct[2])
        attackers = list(set(attackers))
        neighs, _ = get_neigh(adj_mat, collidor)
        non_attackers = list(set(neighs) ^ set(attackers))

        for node in non_attackers:
            adj_mat[collidor, node] = 1
            adj_mat[node, collidor] = -1
    return adj_mat

def orient_extent(tri_nodes, adj_mat):
    if len(tri_nodes) > 0:
        for (i, j, k) in tri_nodes:
            if (adj_mat[i, j] == 1 ) & (adj_mat[j, i] == 0) & (adj_mat[k, j] == 1) & (adj_mat[j, k] == 1) & (adj_mat[i, k] == 0) & (adj_mat[k, i] == 0):
                adj_mat[k, j] = 0
        
        for (i , j, k) in tri_nodes:
            if (adj_mat[i, j] == 1 ) & (adj_mat[j, i] == 0) & (adj_mat[j, k] == 1) & (adj_mat[k, j] == 0) & (adj_mat[i, k] == 1) & (adj_mat[k, i] == 1):
                adj_mat[k, i] = 0

    return adj_mat

def PC(cov_, adj_mat, var_num, n_sample, alpha):
    adj, sepSet = skeleton_discovery(cov_, adj_mat, var_num, n_sample, alpha) 
    adj_mat, v_struct, collidors, tri_nodes = get_v_struct(adj, sepSet, var_num)
    adj_mat = orient_edges(adj_mat, v_struct, collidors)
    adj_mat = orient_extent(tri_nodes, adj_mat)
    return adj_mat

adj = PC(cov_, adj_mat, var_num, n_sample, alpha)

# cg = pc(np.transpose(data), alpha)

