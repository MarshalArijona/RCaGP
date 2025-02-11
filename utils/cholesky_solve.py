import torch.linalg as linalg
'''
solving linear equation given a matrix A and a Cholesky factor L 
'''
def cholesky_solve(B, L, upper=False):
    return linalg.solve_triangular(L.T, linalg.solve_triangular(L, B, upper=False), upper=True)