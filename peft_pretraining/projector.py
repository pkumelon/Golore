import torch

@torch.no_grad()
def get_random_orthogonal_matrix(n, m, dtype = torch.float64):
    # return torch.randn(n, m, dtype = dtype)
    Z = torch.randn(n, m, dtype = torch.float64)
    U, S, Vh = torch.linalg.svd(Z.T @ Z)
    S = 1 / torch.sqrt(S)
    return (Z @ U @ torch.diag(S) @ Vh).to(dtype)

@torch.no_grad()
def get_orthogonal_matrix(weights, rank, type):
    module_params = weights

    if module_params.data.dtype != torch.float:
        float_data = False
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float()
    else:
        float_data = True
        matrix = module_params.data
        
    U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
    
    #make the smaller matrix always to be orthogonal matrix
    if type=='right':
        A = U[:, :rank] @ torch.diag(s[:rank])
        B = Vh[:rank, :]
        
        if not float_data:
            B = B.to(original_device).type(original_type)
        return B
    elif type=='left':
        A = U[:, :rank]
        B = torch.diag(s[:rank]) @ Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
        return A
    elif type=='full':
        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
            B = B.to(original_device).type(original_type)
        return [A, B]
    else:
        raise ValueError('type should be left, right or full')
    

# @torch.no_grad()
# def get_norm_vector(v: torch.Tensor):
#     return v / v.norm()

# @torch.no_grad()
# def get_orthogonal_vector(vec):
#     return vec
#     result = torch.zeros_like(vec)
#     for idx in range(vec.shape[0]):
#         tmp = get_norm_vector(vec[idx].detach().clone())
#         for idx1 in range(idx):
#             tmp -= (tmp * vec[idx1]).sum() * vec[idx1]
#         result[idx] = tmp
#     return result

# @torch.no_grad()
# def get_random_orthogonal_matrix(n, m, dtype = torch.float64):
#     fl = False
#     if n > m:
#         n, m = m, n
#         fl = True
#     result = torch.zeros(n, m, dtype = torch.float64)
#     result[0] = get_norm_vector(torch.randn(1, m, dtype = torch.float64))
#     gauss = torch.zeros(n, m, dtype = torch.float64)
#     gauss[0] = result[0] / result[0][0]
#     base = torch.zeros(m, m, dtype = torch.float64)
#     for idx in range(1, m):
#         base[idx][0] = gauss[0][idx].detach().clone()
#         base[idx][idx] = -1
#     for idx in range(1, n):
#         # print(idx)
#         result[idx] = get_norm_vector(torch.randn(m - idx, dtype = torch.float64) @ get_orthogonal_vector(base[idx:m]))
#         gauss[idx] = result[idx].detach().clone()
#         for idx1 in range(idx):
#             gauss[idx] -= gauss[idx][idx1] * gauss[idx1]
#         tem = gauss[idx][idx].detach().clone()
#         gauss[idx] /= tem
#         for idx1 in range(idx):
#             tem = gauss[idx1][idx] / gauss[idx][idx]
#             gauss[idx1] -= tem * gauss[idx]
#             base[idx + 1:,idx1] -= tem * gauss[idx,idx + 1:]
#         for idx1 in range(idx + 1, m):
#             base[idx1][idx] = gauss[idx][idx1].detach().clone()
#     result = result.to(dtype = dtype)
#     return result.T if fl else result