import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

import pickle


def icp(S_move, S_fix, max_iterations=20, tolerance=1e-6, max_distance=1.0, threshold_iter=20):
    """
    S_move: np.ndarray, shape (m, 2) — новая точка (двумерная)
    S_fix: np.ndarray, shape (m, 2) — предыдущая точка (двумерная)
    """
    src = np.copy(S_move)  # исходная (двигаемая) облако точек
    dst = np.copy(S_fix)  # целевое (фиксированное)
    
    src_n = np.linalg.norm(src, axis=1)
    dst_n = np.linalg.norm(dst, axis=1)
    src = src[src_n > 0.1]
    dst = dst[dst_n > 0.1]

    P = np.copy(src)

    R = np.eye(2)
    t = np.zeros(2)

    R_global = np.eye(2)
    t_global = np.zeros(2)
    tree = KDTree(dst)

    # for i in range(max_iterations):
    #     dist, ind = tree.query(P, k=1)
    #     dist = dist.ravel()
    #     ind = ind.ravel()
    #     if i < threshold_iter:
    #         valid_mask = dist < max_distance
    #         P = P[valid_mask]
    #         matched_dst = dst[ind[valid_mask]]
    #     else:
    #         tree_P = KDTree(P)
    #         _, ind_D = tree_P.query(dst, k=1)
    #         ind_D = ind_D.ravel()

    #         mutual_mask = []
    #         for i_src, j_dst in enumerate(ind):
    #             if dist[i_src] < max_distance and ind_D[j_dst] == i_src:
    #                 mutual_mask.append((i_src, j_dst))

    #         if not mutual_mask:
    #             break

    #         idxs_src, idxs_dst = zip(*mutual_mask)
    #         P = P[list(idxs_src)]
    #         matched_dst = dst[list(idxs_dst)]

    #     if P.shape[0] < 3:
    #         break

    for i in range(max_iterations):
        # 1. Поиск ближайших соседей
        dist, ind = tree.query(P, k=1)
        dist = dist.ravel()
        ind = ind.ravel()
        # matched_dst = dst[ind]

        # Применяем маску по дистанции

        if i > 20:
            distance_mask = dist < max_distance
            # P = P[distance_mask]
            # matched_dst = dst[ind[distance_mask]]
        else:
            distance_mask = dist < 1.0
        P = P[distance_mask]
        matched_dst = dst[ind[distance_mask]]

                
        # 2. Центроиды
        mean_dst = np.mean(matched_dst, axis=0)
        mean_src = np.mean(P, axis=0)

        # 3. Ковариационная матрица
        Cov = (P - mean_src).T @ (matched_dst - mean_dst) 
        # 4. SVD и получение поворота
        U, _, Vh = np.linalg.svd(Cov)
        R = Vh.T @ U.T

        # 5. Смещение
        t = mean_dst - R @ mean_src

        # 6. Применяем трансформацию
        P = (R @ P.T).T + t

        t_global += t
        R_global = R @ R_global

        # Проверка сходимости
        # if np.linalg.norm(src_transformed - src) < tolerance:
        #     break

    plt.clf()
    plt.scatter(dst[:, 0], dst[:, 1], s=20, c='green', alpha=0.5, label='Карта')
    plt.scatter(src[:, 0], src[:, 1], s=5, c='gray', alpha=0.3, label=f'Исходные точки')
    plt.scatter(P[:, 0], P[:, 1], s=20, alpha=0.7, c='blue', label='Итоговый результат')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    # plt.show()

    return t_global, R_global
    


pcd = pickle.load(open('icp_history_rotation.pkl', 'rb'))
# map, g_t, g_R = pickle.load(open('map_400.pkl', 'rb'))
map = np.array(pcd[0])
g_t = np.zeros(2)
g_R = np.eye(2)

for i in range(1, len(pcd)):
    print(i)
    pcd2 = np.array(pcd[i])

    pcd2 = (g_R @ pcd2.T).T + g_t
    # print(pcd2)
    t, R = icp(pcd2, map, max_iterations=100, max_distance=0.01)
    pcd2 = (R @ pcd2.T).T + t

    g_t += t
    g_R = g_R @ R
    map = np.vstack((map, pcd2))
    if i > 40 and i % 5 == 0:
        plt.show()
    # if i == 400:
        # pickle.dump([map, g_t, g_R], open('map_400.pkl', 'wb'))
# pcd1 = pcd[40]
# pcd2 = pcd[55]

# t, R = icp(pcd2, pcd1, max_iterations=100)
