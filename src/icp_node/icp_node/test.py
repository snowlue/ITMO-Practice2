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
        matched_dst = dst[ind.ravel()]

        # 2. Центроиды
        mean_dst = np.mean(matched_dst, axis=0)
        mean_src = np.mean(src, axis=0)

        # 3. Ковариационная матрица
        Cov = np.einsum('ni,nj->ij', matched_dst - mean_dst, src - mean_src)

        # 4. SVD и получение поворота
        U, _, Vt = np.linalg.svd(Cov)
        R = U.T @ Vt
        # R = U @ Vt.T
        # R = U.T @ Vt.T
        # R = U @ Vt

        # 5. Смещение
        t = mean_dst - R @ mean_src

        # 6. Применяем трансформацию
        src_transformed = (R @ src.T).T + t

        # convergence_error = np.linalg.norm(src_transformed - src)
        # if convergence_error < tolerance:
        #     break

        src = src_transformed
    
    plt.clf()
    plt.scatter(src[:, 0], src[:, 1], s=20, alpha=0.7, label='Итоговый результат')
    plt.scatter(S_move[:, 0], S_move[:, 1], s=20, alpha=0.7, label=f'Исходные точки')
    plt.scatter(dst[:, 0], dst[:, 1], s=20, alpha=0.7, label='Целевые точки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    return t, R


pcd = pickle.load(open('icp_history.pkl', 'rb'))
pcd1 = pcd[0]
pcd2 = pcd[1]

t, R = icp(pcd2, pcd1, max_iterations=100)
plt.show()