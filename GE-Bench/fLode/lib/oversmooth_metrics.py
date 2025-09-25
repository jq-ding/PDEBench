import torch

def effective_rank(X, eps=1e-8):
    """
    Compute the effective rank of a feature matrix using PyTorch.
    
    Parameters:
        X: torch.Tensor of shape (N, d), on any device
        eps: float, small constant to avoid log(0)
    
    Returns:
        torch scalar: effective rank
    """
    # Compute singular values
    X = X.real if torch.is_complex(X) else X
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    S = S[S > eps]  # filter out near-zero singular values

    p = S / S.sum()
    entropy = -(p * (p + eps).log()).sum()
    return entropy.exp()


# def class_mix_score(X, y, delta=1e-6, eps=1e-8, X0=None):
#     """
#     Compute the class-mix convergence score S^(l) in PyTorch.
    
#     Parameters:
#         X: torch.Tensor of shape (N, d), feature matrix
#         y: torch.Tensor of shape (N,), integer class labels
#         delta: float, stabilization constant
#         eps: float, to avoid division by 0
#         X0: torch.Tensor or None, initial features X^(0) for normalization
    
#     Returns:
#         torch scalar: normalized score S^(l) if X0 is provided; else rho^(l)
#     """

#     def pairwise_energy(X, y, same_class=True):
#         # Compute pairwise distances
#         diff = X.unsqueeze(1) - X.unsqueeze(0)  # (N, N, d)
#         dist_sq = (diff ** 2).sum(dim=-1)       # (N, N)

#         # Build mask
#         same = (y.unsqueeze(0) == y.unsqueeze(1))  # (N, N)
#         mask = same if same_class else (~same)

#         # Exclude diagonal and duplicate pairs
#         tril_mask = torch.tril(torch.ones_like(mask), diagonal=-1).bool()
#         valid_mask = mask & tril_mask

#         total = dist_sq[valid_mask].sum()
#         count = valid_mask.sum().float()
#         return total / (count + eps)

#     E_w = pairwise_energy(X, y, same_class=True)
#     E_b = pairwise_energy(X, y, same_class=False)
#     rho = E_w / (E_b + eps)

#     if X0 is not None:
#         E0_w = pairwise_energy(X0, y, same_class=True)
#         E0_b = pairwise_energy(X0, y, same_class=False)
#         rho0 = E0_w / (E0_b + eps)
#         S = 1.0 - torch.abs(rho - 1.0) / (torch.abs(rho0 - 1.0) + delta)
#         return S
#     else:
#         return rho
    

def class_mix_score(X, y, delta=1e-6, eps=1e-8, X0=None, distance_type='euclidean'):
    """
    Compute the class-mix convergence score S^(l) in PyTorch.
    
    Parameters:
        X: torch.Tensor of shape (N, d), feature matrix
        y: torch.Tensor of shape (N,), integer class labels
        delta: float, stabilization constant
        eps: float, to avoid division by 0
        X0: torch.Tensor or None, initial features X^(0) for normalization
        distance_type: str, 'euclidean' or 'cosine'
    
    Returns:
        torch scalar: normalized score S^(l) if X0 is provided; else rho^(l)
    """
    X = X.real if torch.is_complex(X) else X
    def pairwise_energy(X, y, same_class=True):
        if distance_type == 'cosine':
            X = torch.nn.functional.normalize(X, dim=1, eps=eps)
            dist = 1 - torch.matmul(X, X.T).clamp(-1 + eps, 1 - eps)  # cosine distance: 1 - cosine similarity
        else:
            X_norm = (X ** 2).sum(dim=1).view(-1, 1)
            dist = X_norm + X_norm.t() - 2 * torch.matmul(X, X.t())
            # dist = torch.clamp(dist, min=0) 

        # Build mask
        same = (y.unsqueeze(0) == y.unsqueeze(1))  # (N, N)
        mask = same if same_class else (~same)

        # Exclude diagonal and duplicate pairs
        tril_mask = torch.tril(torch.ones_like(mask), diagonal=-1).bool()
        valid_mask = mask & tril_mask

        total = dist[valid_mask].sum()
        count = valid_mask.sum().float()
        
        del X_norm, dist, same, mask, tril_mask, valid_mask
        
        return total / (count + eps)

    E_w = pairwise_energy(X, y, same_class=True)
    E_b = pairwise_energy(X, y, same_class=False)
    rho = E_w / (E_b + eps)

    if X0 is not None:
        E0_w = pairwise_energy(X0, y, same_class=True)
        E0_b = pairwise_energy(X0, y, same_class=False)
        rho0 = E0_w / (E0_b + eps)
        S = 1.0 - torch.abs(rho - 1.0) / (torch.abs(rho0 - 1.0) + delta)
        if S>1:
            print("S>1", S)
        return S
    else:
        return rho
