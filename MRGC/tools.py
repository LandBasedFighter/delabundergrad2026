import torch

def pca_svd(X, k=2):
    X_centered = X - X.mean(dim=0)
    U, S, V = torch.svd(X_centered)
    principal_components = V[:, :k]
    X_pca = torch.mm(X_centered, principal_components)
    return X_pca



def neighbor_graph(x, k=None, epsilon=None, symmetrize=True):
    assert (k is None) ^ (epsilon is None)
    dist = torch.cdist(x, x, p=2)
    adj = torch.zeros_like(dist)
    if k is not None:
        _, nn_indices = torch.topk(dist, k + 1, dim=1, largest=False, sorted=False)
        nn_indices = nn_indices[:, 1:]
        for i in range(x.shape[0]):
            adj[i, nn_indices[i]] = 1
            if symmetrize:
                for j in nn_indices[i]:
                    adj[j, i] = 1
    else:
        p_idxs, n_idxs = torch.nonzero(dist <= epsilon, as_tuple=True)
        for p_idx, n_idx in zip(p_idxs, n_idxs):
            if p_idx != n_idx:
                adj[p_idx, n_idx] = 1
                if symmetrize:
                    adj[n_idx, p_idx] = 1
    return adj


def laplacian(W, normed=True, return_diag=False):
    lap = -W 
    lap.fill_diagonal_(0)

    d = -lap.sum(dim=1) 

    if normed:
        d_sqrt = torch.sqrt(d)
        d_zeros = d_sqrt == 0
        d_sqrt[d_zeros] = 1

        lap = lap / d_sqrt[:, None]
        lap = lap / d_sqrt[None, :]
        
        lap.diagonal().copy_(1 - d_zeros.float())
    else:
        lap.diagonal().copy_(d)

    if return_diag:
        return lap, d
    return lap



def wasserstein_distance(m1, m2, dist):
    return torch.sum(torch.abs(m1 - m2) * dist)

def compute_ricci_curvature(adj, alpha=0.5):
    N = adj.size(0)
    
    inf = float('inf')
    dist = torch.full((N, N), inf, dtype=torch.float32, device=adj.device)
    dist[adj > 0] = 1 / adj[adj > 0]

    degree = adj.sum(dim=1, keepdim=True)
    mass = torch.zeros_like(adj, dtype=torch.float32)
    mass.fill_diagonal_(alpha)
    neighbors_mask = adj > 0
    degree_broadcasted = degree.repeat(1, adj.size(1))  # shape: [N, N]
    mass[neighbors_mask] = ((1 - alpha) * adj[neighbors_mask] / degree_broadcasted[neighbors_mask])

    # Wasserstein
    abs_diff_mass = torch.abs(mass.unsqueeze(1) - mass.unsqueeze(0))
    W = torch.sum(abs_diff_mass * dist.unsqueeze(0), dim=-1)
    W[dist == float('inf')] = 0

    # K_R(i, j)
    ricci_curvature = torch.ones((N, N), dtype=torch.float32, device=adj.device)
    mask = dist > 0
    ricci_curvature[mask] -= W[mask] / dist[mask]

    # K_R(i)
    node_ricci = torch.zeros(N, dtype=torch.float32, device=adj.device)
    for i in range(N):
        neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) > 0:
            weights = adj[i, neighbors] / degree[i]
            node_ricci[i] = torch.sum(weights * ricci_curvature[i, neighbors]) 
    
    return node_ricci


def calculate_volume(Z, d=1.0):
    Z_mean = torch.mean(Z, dim=0)
    diff = Z - Z_mean 
    
    # (Z - Z_mean)(Z - Z_mean)^T
    outer_product = torch.mm(diff.T, diff)
    
    # \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T
    scaled_outer_product = (d / Z.size(0)) * outer_product
    
    # I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T
    matrix_sum = torch.eye(Z.size(1), device=Z.device) + scaled_outer_product
    
    # \frac{1}{2} \log_2(I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T)
    det_matrix_sum = torch.det(matrix_sum)
    volume = 0.5 * torch.log2(det_matrix_sum)
    
    return volume



# ====================================================================
import torch

def compute_normal_vector(Z_i, neighbors, k):
    """
    Compute normal vector at point i using k-nearest neighbors
    
    Args:
        Z_i: Point coordinates [d']
        neighbors: Neighbor coordinates [k, d']
        k: Number of neighbors
    Returns:
        normal vector [d']
    """
    # Calculate center of k neighborhoods
    c_i = neighbors.mean(dim=0)  # [d']
    
    # Center the neighbors
    Y = neighbors - c_i.unsqueeze(0)  # [k, d']
    
    # Compute Y^T Y
    YTY = torch.matmul(Y.T, Y)  # [d', d']
    
    # Get eigenvector corresponding to smallest eigenvalue
    eigenvalues, eigenvectors = torch.linalg.eigh(YTY)
    
    # The normal vector is the eigenvector corresponding to smallest eigenvalue
    normal = eigenvectors[:, 0]  # [d']
    
    # Ensure consistent orientation
    if torch.sum(normal * (Z_i - c_i)) < 0:
        normal = -normal
        
    return normal

def project_to_tangent_space(neighbors, center, normal):
    """
    Project neighbors to tangent space
    
    Args:
        neighbors: Neighbor coordinates [k, d']
        center: Center point coordinates [d']
        normal: Normal vector [d']
    Returns:
        projected coordinates matrix O_i [k, d'-1]
    """
    # Center the neighbors
    centered = neighbors - center.unsqueeze(0)  # [k, d']
    
    # Get basis for tangent space (excluding normal direction)
    Q = torch.eye(normal.shape[0], device=normal.device)
    Q = Q - torch.outer(normal, normal)
    Q, _ = torch.linalg.qr(Q)  # [d', d']
    
    # Remove the last vector (corresponding to normal direction)
    basis = Q[:, :-1]  # [d', d'-1]
    
    # Project onto tangent space
    O_i = torch.matmul(centered, basis)  # [k, d'-1]
    
    return O_i


def compute_gaussian_curvature(Z, k):
    """
    Compute Gaussian curvature for all points in Z
    
    Args:
        Z: Point cloud coordinates [n, d']
        k: Number of neighbors
    Returns:
        Gaussian curvature for each point [n]
    """
    n = Z.shape[0]
    curvatures = torch.zeros(n, device=Z.device)
    
    for i in range(n):
        # Get k nearest neighbors
        diffs = Z - Z[i].unsqueeze(0)
        distances = torch.sum(diffs ** 2, dim=1)
        _, indices = torch.topk(distances, k + 1, largest=False)
        neighbors = Z[indices[1:]]  # Exclude the point itself
        
        # Compute normal vector
        normal = compute_normal_vector(Z[i], neighbors, k)
        
        # Project to tangent space
        O_i = project_to_tangent_space(neighbors, Z[i], normal)
        
        # Compute target values (deviation from tangent plane)
        T = torch.sum((neighbors - Z[i].unsqueeze(0)) * normal.unsqueeze(0), dim=1)
        
        # try:
        # k = O_i.shape[0]
        m = O_i.shape[1]
        # O = torch.einsum('ij,ik->ijk', O_i, O_i).reshape(k, m * m)
        O = (O_i.unsqueeze(2) *  O_i.unsqueeze(1)).reshape(k, m * m)
        OTO = O.T @ O
        Theta = (torch.inverse(OTO + torch.eye(OTO.shape[0], device=OTO.device)) @ O.T @ T).reshape(m, m)

        # eigenvalues, eigenvectors = torch.linalg.eigh(Theta)
        # curvatures[i] = torch.det(Theta)

        _, eigenvalues, _ = torch.svd(Theta)
        k1, k2 = eigenvalues[:2]
        curvatures[i] = k1 * k2

        # except:
            # curvatures[i] = 0

    return curvatures



# ====================================================================
# batch
def compute_normals_batch(Z, neighbors):
    """
    Compute normal vectors for all points simultaneously
    
    Args:
        Z: Points [n, d']
        neighbors: Neighbor points [n, k, d']
    Returns:
        normals: [n, d']
    """
    # Calculate centers
    centers = neighbors.mean(dim=1)  # [n, d']
    
    # Center the neighbors
    Y = neighbors - centers.unsqueeze(1)  # [n, k, d']
    
    # Compute Y^T Y for all points
    YTY = torch.bmm(Y.transpose(1, 2), Y)  # [n, d', d']
    
    # Get eigenvectors for all points
    eigenvalues, eigenvectors = torch.linalg.eigh(YTY)  # [n, d'], [n, d', d']
    
    # Take eigenvector corresponding to smallest eigenvalue
    normals = eigenvectors[:, :, 0]  # [n, d']
    
    # Ensure consistent orientation
    dots = torch.sum(normals * (Z - centers), dim=1)  # [n]
    normals = normals * torch.sign(dots).unsqueeze(-1)  # [n, d']
    
    return normals

def project_to_tangent_space_batch(neighbors, centers, normals):
    """
    Project neighbors to tangent spaces for all points simultaneously
    
    Args:
        neighbors: [n, k, d']
        centers: [n, d']
        normals: [n, d']
    Returns:
        projected: [n, k, d'-1]
    """
    n, k, d = neighbors.shape
    
    # Center the neighbors
    centered = neighbors - centers.unsqueeze(1)  # [n, k, d']
    
    # Create orthogonal bases for tangent spaces
    I = torch.eye(d, device=neighbors.device).unsqueeze(0).expand(n, -1, -1)  # [n, d', d']
    Q = I - normals.unsqueeze(2) @ normals.unsqueeze(1)  # [n, d', d']
    Q = torch.linalg.qr(Q).Q  # [n, d', d']
    
    # Remove last vector (corresponding to normal direction)
    basis = Q[:, :, :-1]  # [n, d', d'-1]
    
    # Project onto tangent spaces
    projected = torch.bmm(centered, basis)  # [n, k, d'-1]
    
    return projected

def compute_curvature_batch(O, T):
    """
    Compute curvature for all points simultaneously
    
    Args:
        O: Projected points [n, k, d-1]
        T: Target values [n, k]
    Returns:
        curvatures: [n]
    """
    n, k, m = O.shape
    
    # Reshape O for batch matrix multiplication
    O_reshaped = (O.unsqueeze(3) * O.unsqueeze(2)).reshape(n, k, m * m)  # [n, k, m*m]
    
    # Compute OTO and its inverse for all points
    OTO = torch.bmm(O_reshaped.transpose(1, 2), O_reshaped)  # [n, m*m, m*m]
    reg = torch.eye(m * m, device=O.device).unsqueeze(0).expand(n, -1, -1)
    OTO_inv = torch.inverse(OTO + reg)  # [n, m*m, m*m]
    
    # Compute Theta for all points
    Theta = torch.bmm(torch.bmm(OTO_inv, O_reshaped.transpose(1, 2)), 
                     T.unsqueeze(2))  # [n, m*m, 1]
    Theta = Theta.reshape(n, m, m)  # [n, m, m]
    
    # Compute SVD for all matrices
    _, S, _ = torch.svd(Theta)  # [n, m], assuming m is the smaller dimension
    
    # Compute curvature as product of first two singular values
    curvatures = S[:, 0] * S[:, 1]  # [n]
    
    return curvatures

def compute_gaussian_curvature_batch(Z, k):
    """
    Compute Gaussian curvature for all points simultaneously
    
    Args:
        Z: Point cloud [n, d']
        k: Number of neighbors
    Returns:
        curvatures: [n]
    """
    n = Z.shape[0]
    
    # Compute all pairwise distances
    distances = torch.cdist(Z, Z)  # [n, n]
    
    # Get k+1 nearest neighbors (including self)
    _, indices = torch.topk(-distances, k + 1, dim=1)  # [n, k+1]
    
    # Get neighbor points (excluding self)
    neighbors = Z[indices[:, 1:]]  # [n, k, d']
    
    # Compute normal vectors
    normals = compute_normals_batch(Z, neighbors)  # [n, d']
    
    # Project to tangent space
    O = project_to_tangent_space_batch(neighbors, Z, normals)  # [n, k, d'-1]
    
    # Compute target values (deviation from tangent plane)
    T = torch.sum((neighbors - Z.unsqueeze(1)) * normals.unsqueeze(1), dim=2)  # [n, k]
    
    # Compute curvatures
    curvatures = compute_curvature_batch(O, T)  # [n]
    
    return curvatures