import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CONVERSIÓN DE RODRIGUES (VECTOR A MATRIZ)
def rodrigues_vec_to_mat(rvec):
    rvec = np.asarray(rvec, dtype=float).flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-08:
        return np.eye(3)
    r = rvec / theta
    kx, ky, kz = r
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

# CONVERSIÓN DE RODRIGUES (MATRIZ A VECTOR)
def rodrigues_mat_to_vec(R):
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-08:
        return np.zeros(3)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    r = np.array([rx, ry, rz])
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-08:
        M = (R + np.eye(3)) / 2.0
        col = np.argmax(np.diag(M))
        v = M[:, col]
        v = v / np.linalg.norm(v)
        return v * theta
    return r / norm_r * theta

# CÁLCULO DE JACOBIANO NUMÉRICO
def numerical_jacobian(func, x0, args=(), eps=1e-06):
    f0 = func(x0, *args)
    J = np.zeros((len(f0), len(x0)))
    for i in range(len(x0)):
        x_step = x0.copy()
        x_step[i] += eps
        f_step = func(x_step, *args)
        J[:, i] = (f_step - f0) / eps
    return (J, f0)

# OPTIMIZACIÓN NO LINEAL (LEVENBERG-MARQUARDT)
def custom_least_squares_lm(func, x0, args=(), max_iter=50, tol=1e-05):
    x = x0.copy()
    lam = 0.001
    J, f = numerical_jacobian(func, x, args)
    cost = np.sum(f ** 2)
    for i in range(max_iter):
        H = J.T @ J
        diag_H = np.maximum(np.diag(H), 1e-08)
        A = H + lam * np.diag(diag_H)
        g = -J.T @ f
        try:
            delta = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(delta) < tol * (np.linalg.norm(x) + tol):
            break
        x_new = x + delta
        f_new = func(x_new, *args)
        cost_new = np.sum(f_new ** 2)
        if cost_new < cost:
            x = x_new
            f = f_new
            cost = cost_new
            lam /= 10.0
            J, _ = numerical_jacobian(func, x, args)
        else:
            lam *= 10.0

    class OptResult:
        pass
    res = OptResult()
    res.x = x
    res.cost = cost / 2.0
    return res
world_points_dict = {20: np.array([[550, 550], [650, 550], [650, 650], [550, 650]], dtype=float), 21: np.array([[2350, 550], [2450, 550], [2450, 650], [2350, 650]], dtype=float), 22: np.array([[550, 1350], [650, 1350], [650, 1450], [550, 1450]], dtype=float), 23: np.array([[2350, 1350], [2450, 1350], [2450, 1450], [2350, 1450]], dtype=float)}
ids_tablero = [20, 21, 22, 23]

# CÁLCULO DE HOMOGRAFÍA (MÉTODO DLT)
def compute_homography_dlt(pts_src, pts_dst):
    pts_src = np.asarray(pts_src, dtype=float)
    pts_dst = np.asarray(pts_dst, dtype=float)
    N = pts_src.shape[0]
    mean_src = np.mean(pts_src, axis=0)
    dist_src = np.linalg.norm(pts_src - mean_src, axis=1)
    scale_src = np.sqrt(2) / np.mean(dist_src)
    T_src = np.array([[scale_src, 0, -scale_src * mean_src[0]], [0, scale_src, -scale_src * mean_src[1]], [0, 0, 1]])
    pts_src_norm = np.ones((N, 3))
    pts_src_norm[:, :2] = pts_src
    pts_src_norm = (T_src @ pts_src_norm.T).T[:, :2]
    mean_dst = np.mean(pts_dst, axis=0)
    dist_dst = np.linalg.norm(pts_dst - mean_dst, axis=1)
    scale_dst = np.sqrt(2) / np.mean(dist_dst)
    T_dst = np.array([[scale_dst, 0, -scale_dst * mean_dst[0]], [0, scale_dst, -scale_dst * mean_dst[1]], [0, 0, 1]])
    pts_dst_norm = np.ones((N, 3))
    pts_dst_norm[:, :2] = pts_dst
    pts_dst_norm = (T_dst @ pts_dst_norm.T).T[:, :2]
    A = []
    for i in range(N):
        x, y = pts_src_norm[i]
        u, v = pts_dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    H_norm = Vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[-1, -1]
    return H

def calcular_K_del_video(ruta_video):
    print('\n' + '=' * 60)
    print('CALIBRANDO MATRIZ K A PARTIR DEL VIDEO (TODOS LOS FRAMES)')
    print('=' * 60)
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print('Error: No se pudo abrir el video')
        return None
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Resolución video: {ancho}x{alto}')
    print(f'Total frames: {total_frames}')
    diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parametros_aruco = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario_aruco, parametros_aruco)
    homografias = []
    frames_calib = []
    errores_calib = []
    all_world_pts = []
    all_img_pts = []
    frame_actual = 0
    PASO_FRAMES = 30
    print(f'Recorriendo el vídeo para acumular homografías (1 de cada {PASO_FRAMES} frames)...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_actual % PASO_FRAMES != 0:
            frame_actual += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = detector.detectMarkers(gray)
        marcadores_encontrados = {}
        if ids is not None:
            for i, aruco_id in enumerate(ids.flatten()):
                if aruco_id in ids_tablero:
                    marcadores_encontrados[aruco_id] = esquinas[i][0]
        if len(marcadores_encontrados) == 4:
            image_points = []
            world_points = []
            for aruco_id in ids_tablero:
                image_points.extend(marcadores_encontrados[aruco_id])
                world_points.extend(world_points_dict[aruco_id])
            world_pts_arr = np.array(world_points, dtype=float)
            img_pts_arr = np.array(image_points, dtype=float)
            H = compute_homography_dlt(world_pts_arr, img_pts_arr)
            world_h = np.column_stack([world_pts_arr, np.ones(len(world_pts_arr))])
            proj_h = (H @ world_h.T).T
            proj_h = proj_h[:, :2] / proj_h[:, 2:3]
            error_h = float(np.mean(np.linalg.norm(img_pts_arr - proj_h, axis=1)))
            homografias.append(H)
            frames_calib.append(frame_actual)
            errores_calib.append(error_h)
            all_world_pts.append(world_pts_arr)
            all_img_pts.append(img_pts_arr)
            print(f'  Frame {frame_actual}: Homografía #{len(homografias)} calculada  (error H: {error_h:.2f}px)')
        frame_actual += 1
    cap.release()
    if len(homografias) < 4:
        print(f'ERROR: Se necesitan al menos 4 homografías. Solo se encontraron {len(homografias)}.')
        return None
    print(f'\nTotal homografías acumuladas: {len(homografias)}')
    print('Calculando K (incluyendo cx, cy algebraicos) mediante método Zhang completo...')

    # MÉTODO DE ZHANG: EXTRACCIÓN ALGEBRAICA DE K
    def vij(H, i, j):
        hi = H[:, i]
        hj = H[:, j]
        return np.array([hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1], hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]])
    V = []
    for H in homografias:
        v12 = vij(H, 0, 1)
        v11 = vij(H, 0, 0)
        v22 = vij(H, 1, 1)
        V.append(v12)
        V.append(v11 - v22)
    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]
    B11, B12, B22, B13, B23, B33 = b
    V5 = V[:, [0, 2, 3, 4, 5]]
    _, _, Vh5 = np.linalg.svd(V5)
    b5 = Vh5[-1]
    B11, B22, B13, B23, B33 = b5
    B12 = 0.0
    denom = B11 * B22 - B12 ** 2
    cy = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 ** 2 + cy * (B12 * B13 - B11 * B23)) / B11
    fx_sq = lam / B11
    fy_sq = lam * B22 / denom
    cx = -B13 * fx_sq / lam
    fx = np.sqrt(abs(fx_sq))
    fy = np.sqrt(abs(fy_sq))
    if not 100 < fx < 100000:
        print(f'[!] fx={fx:.1f} fuera de rango. Verificar puntos mundo e imagen.')
    if not 100 < fy < 100000:
        print(f'[!] fy={fy:.1f} fuera de rango. Verificar puntos mundo e imagen.')
    print(f'  fx = {fx:.2f} px')
    print(f'  fy = {fy:.2f} px')
    print(f'  cx = {cx:.2f} px  (calculado, imagen ancho={ancho})')
    print(f'  cy = {cy:.2f} px  (calculado, imagen alto={alto})')
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    print('\n' + '=' * 60)
    print('--- FASE 5: OPTIMIZACIÓN NO LINEAL (Levenberg-Marquardt) ---')
    n_views = len(homografias)
    intrinsics_init = np.array([fx, fy, cx, cy, 0.0, 0.0])
    extrinsics_init = []
    K_inv = np.linalg.inv(K)
    for H in homografias:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lam = 1.0 / np.linalg.norm(K_inv @ h1)
        r1 = lam * (K_inv @ h1)
        r2 = lam * (K_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lam * (K_inv @ h3)
        R_approx = np.column_stack((r1, r2, r3))
        U_rot, _, Vh_rot = np.linalg.svd(R_approx)
        R_valid = U_rot @ Vh_rot
        if np.linalg.det(R_valid) < 0:
            R_valid = -R_valid
        rvec = rodrigues_mat_to_vec(R_valid)
        extrinsics_init.extend(rvec.flatten())
        extrinsics_init.extend(t.flatten())
    x0 = np.concatenate((intrinsics_init, extrinsics_init))

    # MÉTODO DE ZHANG: REFINAMIENTO MEDIANTE ERROR DE REPROYECCIÓN
    def reprojection_residuals(params, n_views, all_world_pts, all_img_pts):
        f_x, f_y, c_x, c_y, k1, k2 = params[:6]
        extrinsics = params[6:].reshape((n_views, 6))
        res = []
        for i in range(n_views):
            rvec = extrinsics[i, :3]
            tvec = extrinsics[i, 3:]
            R = rodrigues_vec_to_mat(rvec)
            pts_2d = all_world_pts[i]
            pts_3d = np.column_stack((pts_2d, np.zeros(len(pts_2d))))
            pts_cam = (R @ pts_3d.T).T + tvec
            x = pts_cam[:, 0] / pts_cam[:, 2]
            y = pts_cam[:, 1] / pts_cam[:, 2]
            r2 = x ** 2 + y ** 2
            dist_factor = 1.0 + k1 * r2 + k2 * r2 ** 2
            x_dist = x * dist_factor
            y_dist = y * dist_factor
            u = f_x * x_dist + c_x
            v = f_y * y_dist + c_y
            proj_pts = np.column_stack((u, v))
            error = (proj_pts - all_img_pts[i]).flatten()
            res.extend(error)
        return np.array(res)
    print(f'Optimizando {len(x0)} parámetros ({n_views} vistas)... (puede tardar un momento)')
    opt_res = custom_least_squares_lm(reprojection_residuals, x0, args=(n_views, all_world_pts, all_img_pts), max_iter=50)
    x_opt = opt_res.x
    fx_opt, fy_opt, cx_opt, cy_opt, k1_opt, k2_opt = x_opt[:6]
    K_opt = np.array([[fx_opt, 0.0, cx_opt], [0.0, fy_opt, cy_opt], [0.0, 0.0, 1.0]])
    print('\n[Fase 5 Completada]')
    print(f'Mejora del error total de reproyección: {opt_res.cost:.2f} (final)')
    print(f'Coeficientes de distorsión calculados: k1={k1_opt:.5f}, k2={k2_opt:.5f}')
    datos_calibracion = {'frames': frames_calib, 'errores_H': errores_calib, 'k1': k1_opt, 'k2': k2_opt}
    return (K_opt, datos_calibracion)

def plot_calibracion_por_frame(frames, errores_H):
    frames = np.array(frames)
    errores = np.array(errores_H)
    if len(errores) == 0:
        print('No hay datos de calibración para graficar.')
        return
    q1, q3 = np.percentile(errores, [25, 75])
    iqr = q3 - q1
    umbral = np.median(errores) + 1.5 * iqr
    inliers_mask = errores <= umbral
    outliers_mask = ~inliers_mask
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(frames[outliers_mask], errores[outliers_mask], color='red', marker='x', s=80, linewidths=2, label=f'Outliers descartados ({np.sum(outliers_mask)})', zorder=3)
    ax.scatter(frames[inliers_mask], errores[inliers_mask], color='steelblue', alpha=0.8, s=50, edgecolors='black', linewidths=0.5, label=f'Inliers usados en K ({np.sum(inliers_mask)})', zorder=3)
    media_inliers = float(np.mean(errores[inliers_mask]))
    ax.axhline(media_inliers, color='green', linestyle='--', linewidth=2, label=f'Error medio inliers: {media_inliers:.2f} px')
    ax.axhline(umbral, color='orange', linestyle=':', linewidth=2, label=f'Umbral outlier: {umbral:.2f} px')
    ax.set_title('Calidad de Homografías por Frame (Calibración Zhang)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Frame', fontsize=12)
    ax.set_ylabel('Error de Reproyección de Homografía (px)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_video = os.path.join(ruta_script, 'VideoVision.mp4')
    resultado = calcular_K_del_video(ruta_video)
    if resultado is None:
        print('Error: No se pudo calcular K del video')
        exit()
    K_matrix, datos_calibracion = resultado
    print('\n' + '=' * 60)
    print('MATRIZ INTRÍNSECA K CALCULADA:')
    print('=' * 60)
    print(K_matrix)
    print(f'  fx = {K_matrix[0, 0]:.4f} px')
    print(f'  fy = {K_matrix[1, 1]:.4f} px')
    print(f'  cx = {K_matrix[0, 2]:.4f} px')
    print(f'  cy = {K_matrix[1, 2]:.4f} px')
    print('=' * 60 + '\n')
    archivo_salida = os.path.join(ruta_script, 'parametros_camara.npz')
    np.savez(archivo_salida, K=K_matrix, k1=datos_calibracion['k1'], k2=datos_calibracion['k2'], frames=datos_calibracion['frames'], errores_H=datos_calibracion['errores_H'])
    print(f'Parámetros de cámara guardados en {archivo_salida}')
    plot_calibracion_por_frame(datos_calibracion['frames'], datos_calibracion['errores_H'])