import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

import pickle


def icp(S_move, S_fix, max_iterations=20, tolerance=1e-6):
    """
    S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
    S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
    """
    src = np.copy(S_move)  # исходная (двигаемая) облако точек D p
    dst = np.copy(S_fix)  # целевое (фиксированное) M q
    
    R = np.eye(2)
    t = np.zeros(2)

    tree = KDTree(dst)
    
    for iteration in range(max_iterations):
        # 1. Поиск ближайших соседей
        dist, ind = tree.query(src, k=1)
        dist = dist.ravel()
        ind = ind.ravel()
            
        # Применяем маску по дистанции
        distance_mask = dist < 0.1

        src = src[distance_mask]
        matched_dst = dst[ind[distance_mask]]

        # 2. Центроиды
        mean_dst = np.mean(matched_dst, axis=0)
        mean_src = np.mean(src, axis=0)

        # 3. Ковариационная матрица
        Cov = np.einsum('ni,nj->ij', src - mean_src, matched_dst - mean_dst)

        # 4. SVD и получение поворота
        U, _, V = np.linalg.svd(Cov)
        R = V @ U.T

        # print(U @ U.T, V @ V.T, Cov, U @ np.diag(_) @ V.T, sep='\n')
        # print(dist, end='\n')

        # 5. Смещение
        t = mean_dst - R @ mean_src

        # 6. Применяем трансформацию
        src_transformed = (R @ src.T).T + t

        # convergence_error = np.linalg.norm(src_transformed - src)
        if np.linalg.norm(matched_dst - src_transformed) < tolerance:
                break

        src = src_transformed
    
    plt.clf()
    plt.scatter(dst[:, 0], dst[:, 1], s=20, c='green', alpha=0.7, label='Целевые точки')
    plt.scatter(S_move[:, 0], S_move[:, 1], s=20, c='gray', alpha=0.3, label=f'Исходные точки')
    plt.scatter(src[:, 0], src[:, 1], s=20, alpha=0.7, c='b', label='Итоговый результат')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    return t, R


pcd = pickle.load(open('icp_history.pkl', 'rb'))
# for i in range(710, len(pcd)):
#     print(i)
#     pcd1 = np.array(pcd[i - 1])
#     pcd2 = np.array(pcd[i])
#     t, R = icp(pcd2, pcd1, max_iterations=100)
#     plt.show()
pcd1 = pcd[720]
pcd2 = pcd[732]

t, R = icp(pcd2, pcd1, max_iterations=10)
plt.show()