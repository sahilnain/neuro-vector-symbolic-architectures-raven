import torch as t
import torch.nn.functional as F
import numpy as np

def cyclic_column_matrix(c, L, direction="down"):
    c = np.asarray(c)
    N = c.size
    shifts = np.arange(L + 1)
    if direction == "down":
        # A[i, s] = c[(i - s) % N]
        idx = (np.arange(N)[:, None] - shifts[None, :]) % N
    elif direction == "up":
        # A[i, s] = c[(i + s) % N]
        idx = (np.arange(N)[:, None] + shifts[None, :]) % N
    else:
        raise ValueError("direction must be 'down' or 'up'")
    return c[idx]

def mapping_function(A, B, flip_B=True):
    C = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=A.dtype)
    for j in range(A.shape[0]):
        for i in range(A.shape[1]):
            B_cyclic = cyclic_column_matrix(np.flip(B[j,i,:]) if flip_B else B[j,i,:], A.shape[2]-1, direction="down")
            C_partial = A[j, i, :].dot(B_cyclic)
            if(flip_B):
                C_partial = np.roll(C_partial, -1)
            C[j, i, :] = C_partial
    return C

def binding_circular(A, B, alpha=1):
    """
    Binds two block codes vectors by blockwise cicular convolution. 

    Parameters
    ----------
    A: torch FloatTensor (_, _k, _l)
        input vector 1
    B: torch FloatTensor  (_, _k, _l)
        input vector 2
    alpha: int, optional
        specifies multiplicative factor for number of shifts (Default value '1')
    Returns
    -------
    C: torch FloatTensor (_k)
        k-dimensional offset vector that is the result of the binding operation.
    """
    ndim = A.dim()
    # add batch dimension (1) if not there yet
    if ndim==2: 
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
    
    batchSize,k,l = A.shape
    
    # prepare inputs
    A = t.unsqueeze(A,1) # input
    B = t.unsqueeze(B,2) # filter weigths
    B = t.flip(B, [3]) # flip input
    B = t.roll(B, 1, dims=3) # roll by one to fit addition

    # reshape for single CONV
    A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
    B = t.reshape(B, (B.shape[0]*B.shape[1], B.shape[2], B.shape[3]))

    # calculate C = t.remainder(B+A*alpha, self._L)
    C = F.conv1d(F.pad(A, pad=(0,l-1), mode='circular'), B, groups=k*batchSize)
    
    C = t.reshape(C, (batchSize, k, l))

    # Remove batch dimension if it was not there intially
    if ndim==2: 
        C = C.squeeze(0)
    return C

def inv_binding_circular(C,A, alpha=1):
    """
    Inverse binding of two block codes vectors by blockwise cicular correlation. 

    Parameters
    ----------
    A: torch FloatTensor (_, _k, _l)
        input vector 1
    B: torch FloatTensor  (_, _k, _l)
        input vector 2
    alpha: int, optional
        specifies multiplicative factor for number of shifts (Default value '1')
    Returns
    -------
    C: torch FloatTensor (_k)
        k-dimensional offset vector that is the result of the binding operation.
    """

    ndim = A.dim()
    # add batch dimension (1) if not there yet
    if ndim==2: 
        A = A.unsqueeze(0)
        C = C.unsqueeze(0)
    batchSize,k,l = A.shape

    A = t.unsqueeze(A,1) # input
    C = t.unsqueeze(C,2) # filter weigths

    A = t.reshape(A, (1, A.shape[0]*A.shape[2], A.shape[3]))
    C = t.reshape(C, (C.shape[0]*C.shape[1], C.shape[2], C.shape[3]))
        
    B = F.conv1d(F.pad(A, pad=(0,l-1), mode='circular'), C, groups=k*batchSize)
    B = t.reshape(B, (batchSize, k, l))
        
    B = t.flip(B, [2]) # flip input
    B = t.roll(B, 1, dims=2) # roll by one to fit addition

    # Remove batch dimension if it was not there intially
    if ndim==2: 
        B = B.squeeze(0)

    return B

def match_prob(x,y, act=t.nn.Identity()): 
    '''
    Compute similarity between two block codes vectors

    Parameters
    ----------
    x: torch FloatTensor (B,k,l) 
        input vector 1
    y: complex vector (B,k,l)
        input vector 2 
    Output
    ------
    sim: torch FloatTensor (B,)
        output similarity 
    '''
    _,k,l = x.shape
    sim = 1/k*t.sum(x*y, dim=(1,2))
    sim = act(sim)
    return sim

def cosine2pmf(sim,act="Softmax",s=40): 
    # Infer PMF from the similarities (e.g., cosine) 
    if act == "Softmax": 
        out = t.nn.functional.softmax(sim.view(1,-1)*s, dim=-1)
    elif act == "Identity": 
        out = t.nn.functional.normalize(sim.view(1,-1), p=1, dim = -1)
    return out

def main():
    # A = t.tensor(np.random.randint(2, size=(2, 2, 16)) - 1, dtype=t.float)
    # B = t.tensor(np.random.randint(2, size=(2, 2, 16)) - 1, dtype=t.float)
    A = t.zeros((1, 1, 16))
    B = t.zeros((1, 1, 16))
    A[0, 0, 1] = t.tensor((1), dtype=t.float)
    B[0, 0, 2] = t.tensor((1), dtype=t.float)
    C = binding_circular(A, B)
    E = inv_binding_circular(A, B)
    D = mapping_function(A.numpy(), B.numpy(), flip_B=True)
    F = mapping_function(A.numpy(), B.numpy(), flip_B=False)
    # D = inv_binding_circular(A, B)
    # match_prob(C, D)
    print("Convolution:")
    print(C)
    print("Mapping Function (flip_B=True):")
    print(D)
    print("Correlation:")
    print(E)
    print("Mapping Function (flip_B=False):")
    print(F)

    
if __name__ == "__main__":
    main()