import torch

class Hybrid_projector:
    def __init__(self, rank, r0 = None, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        if r0 == None: r0 = rank
        self.r0 = r0
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter, rand = False):
        matrix = None
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.proj_type = 'right'
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right', rand = rand)
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                self.proj_type = 'left'
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left', rand = rand)
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.proj_type = 'left'
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left', rand = rand)
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type),full_rank_grad)
            else:
                self.proj_type = 'right'
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right', rand = rand)
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right', rand = rand)
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left', rand = rand)
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                matrix = self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full', rand = rand)
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)
        return low_rank_grad, matrix
    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device.type)


        return full_rank_grad * self.scale

    @torch.no_grad()
    def get_random_orthogonal_matrix(self, n, m):
        # return torch.randn(n, m, dtype = dtype)
        Z = torch.randn(n, m)
        U, S, Vh = torch.linalg.svd(Z.T @ Z)
        S = 1 / torch.sqrt(S)
        return (Z @ U @ torch.diag(S) @ Vh)

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type, rand = False):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        
        if not rand:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
            idx = torch.multinomial(s, rank).sort().values

            #make the smaller matrix always to be orthogonal matrix
            if type=='right':
                B = Vh[idx, :]
                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type=='left':
                A = U[:, idx]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                return A
            elif type=='full':
                A = U[:, idx]
                B = Vh[idx, :]
                if not float_data:
                    A = A.to(original_device).type(original_type)
                    B = B.to(original_device).type(original_type)
                return [A, B]
            else:
                raise ValueError('type should be left, right or full')
        else:
            if type=='right':
                B = self.get_random_orthogonal_matrix(rank, weights.shape[1])
                if not float_data:
                    B = B.to(original_device).type(original_type)
                return B
            elif type=='left':
                A = self.get_random_orthogonal_matrix(weights.shape[0], rank)
                if not float_data:
                    A = A.to(original_device).type(original_type)
                return A
