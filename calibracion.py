import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
K_matrix = None
world_points_dict = {
    20: np.array([[550, 550], [650, 550], [650, 650], [550, 650]], dtype=float),
    21: np.array([[2350, 550], [2450, 550], [2450, 650], [2350, 650]], dtype=float),
    22: np.array([[550, 1350], [650, 1350], [650, 1450], [550, 1450]], dtype=float),
    23: np.array([[2350, 1350], [2450, 1350], [2450, 1450], [2350, 1450]], dtype=float)
}
ids_tablero = [20, 21, 22, 23]
# CÁLCULO DE HOMOGRAFÍA (MÉTODO DLT)
def compute_homography_dlt(pts_src, pts_dst):
    pts_src = np.asarray(pts_src, dtype=float)
    pts_dst = np.asarray(pts_dst, dtype=float)
    N = pts_src.shape[0]
    mean_src = np.mean(pts_src, axis=0)
    dist_src = np.linalg.norm(pts_src - mean_src, axis=1)
    scale_src = np.sqrt(2) / np.mean(dist_src)
    T_src = np.array([[scale_src, 0, -scale_src * mean_src[0]],
                      [0, scale_src, -scale_src * mean_src[1]],
                      [0, 0, 1]])
    pts_src_norm = np.ones((N, 3))
    pts_src_norm[:, :2] = pts_src
    pts_src_norm = (T_src @ pts_src_norm.T).T[:, :2]
    mean_dst = np.mean(pts_dst, axis=0)
    dist_dst = np.linalg.norm(pts_dst - mean_dst, axis=1)
    scale_dst = np.sqrt(2) / np.mean(dist_dst)
    T_dst = np.array([[scale_dst, 0, -scale_dst * mean_dst[0]],
                      [0, scale_dst, -scale_dst * mean_dst[1]],
                      [0, 0, 1]])
    pts_dst_norm = np.ones((N, 3))
    pts_dst_norm[:, :2] = pts_dst
    pts_dst_norm = (T_dst @ pts_dst_norm.T).T[:, :2]
    A = []
    for i in range(N):
        x, y = pts_src_norm[i]
        u, v = pts_dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    H_norm = Vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[-1, -1]
    return H
# HOMOGRAFÍA ROBUSTA (MÉTODO RANSAC)
def ransac_homografia(pts_src, pts_dst, iteraciones=1000, umbral_error=2.0):
    N = len(pts_src)
    mejor_inliers = 0
    H_best = None
    inliers_mask_best = None
    np.random.seed(42)
    for iter_num in range(iteraciones):
        indices_aleatorios = np.random.choice(N, size=4, replace=False)
        pts_src_sample = pts_src[indices_aleatorios]
        pts_dst_sample = pts_dst[indices_aleatorios]
        try:
            H_candidata = compute_homography_dlt(pts_src_sample, pts_dst_sample)
        except:
            continue
        pts_src_homog = np.ones((N, 3))
        pts_src_homog[:, :2] = pts_src
        pts_dst_proyectados_homog = (H_candidata @ pts_src_homog.T).T
        pts_dst_proyectados = pts_dst_proyectados_homog[:, :2] / pts_dst_proyectados_homog[:, 2:3]
        errores = np.linalg.norm(pts_dst - pts_dst_proyectados, axis=1)
        inliers_mask = errores < umbral_error
        num_inliers = np.sum(inliers_mask)
        if num_inliers > mejor_inliers:
            mejor_inliers = num_inliers
            H_best = H_candidata
            inliers_mask_best = inliers_mask
    if mejor_inliers >= 4:
        pts_src_inliers = pts_src[inliers_mask_best]
        pts_dst_inliers = pts_dst[inliers_mask_best]
        H_best = compute_homography_dlt(pts_src_inliers, pts_dst_inliers)
    return H_best, inliers_mask_best, mejor_inliers
# CÁLCULO DE ERROR DE REPROYECCIÓN
def calcular_error_reproyeccion(world_points, image_points, K, R, t):
    world_pts_2d = np.array(world_points, dtype=float)
    image_pts = np.array(image_points, dtype=float)
    world_pts_3d = np.column_stack([world_pts_2d, np.zeros(len(world_pts_2d))])
    pts_cam = (R @ world_pts_3d.T + t[:, np.newaxis]).T
    pts_img_homogeneos = (K @ pts_cam.T).T
    pts_img_proyectados = pts_img_homogeneos[:, :2] / pts_img_homogeneos[:, 2:3]
    errores = np.linalg.norm(image_pts - pts_img_proyectados, axis=1)
    error_medio = np.mean(errores)
    error_max = np.max(errores)
    return errores, error_medio, error_max, pts_img_proyectados

def calcular_error_homografia(pts_src, pts_dst, H):
    pts_src = np.asarray(pts_src, dtype=float)
    pts_dst = np.asarray(pts_dst, dtype=float)
    pts_src_homog = np.ones((len(pts_src), 3))
    pts_src_homog[:, :2] = pts_src
    pts_dst_proyectados_homog = (H @ pts_src_homog.T).T
    pts_dst_proyectados = pts_dst_proyectados_homog[:, :2] / pts_dst_proyectados_homog[:, 2:3]
    errores = np.linalg.norm(pts_dst - pts_dst_proyectados, axis=1)
    return errores, float(np.mean(errores)), float(np.max(errores))

def ransac_linea_simple(x, y, iteraciones=1000, umbral_residual=0.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_puntos = len(x)
    if n_puntos < 2:
        return None, None

    mejor_inliers = 0
    mejor_mask = None
    mejor_modelo = None
    rng = np.random.default_rng(42)

    for _ in range(iteraciones):
        i1, i2 = rng.choice(n_puntos, size=2, replace=False)
        if np.isclose(x[i1], x[i2]):
            continue

        pendiente = (y[i2] - y[i1]) / (x[i2] - x[i1])
        intercepto = y[i1] - pendiente * x[i1]
        residuales = np.abs(y - (pendiente * x + intercepto))
        mascara = residuales < umbral_residual
        num_inliers = int(np.sum(mascara))

        if num_inliers > mejor_inliers:
            mejor_inliers = num_inliers
            mejor_mask = mascara
            mejor_modelo = (pendiente, intercepto)

    if mejor_mask is None or mejor_inliers < 2:
        pendiente, intercepto = np.polyfit(x, y, 1)
        mejor_mask = np.ones(n_puntos, dtype=bool)
        return (pendiente, intercepto), mejor_mask

    pendiente, intercepto = np.polyfit(x[mejor_mask], y[mejor_mask], 1)
    mejor_modelo = (pendiente, intercepto)
    return mejor_modelo, mejor_mask

def plot_ransac_result(pts_src, pts_dst, H, inliers_mask, umbral_error):
    pts_src = np.asarray(pts_src, dtype=float)
    pts_dst = np.asarray(pts_dst, dtype=float)
    pts_src_homog = np.ones((len(pts_src), 3))
    pts_src_homog[:, :2] = pts_src
    pts_dst_proyectados_homog = (H @ pts_src_homog.T).T
    pts_dst_proyectados = pts_dst_proyectados_homog[:, :2] / pts_dst_proyectados_homog[:, 2:3]
    observed = pts_dst[:, 0]
    predicted = pts_dst_proyectados[:, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(observed)):
        ax.plot([observed[i], predicted[i]], [observed[i], predicted[i]], color='gray', alpha=0.25, linewidth=1)

    ax.scatter(observed[inliers_mask], predicted[inliers_mask], color='steelblue', s=70, edgecolors='black', linewidths=0.6, label=f'Inliers ({np.sum(inliers_mask)})')
    if np.any(~inliers_mask):
        ax.scatter(observed[~inliers_mask], predicted[~inliers_mask], color='red', marker='x', s=80, linewidths=2, label=f'Outliers ({np.sum(~inliers_mask)})')
    else:
        ax.text(0.02, 0.02, 'Sin outliers en este frame', transform=ax.transAxes, fontsize=10, va='bottom', ha='left', bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    lim_inf = min(observed.min(), predicted.min())
    lim_sup = max(observed.max(), predicted.max())
    ax.plot([lim_inf, lim_sup], [lim_inf, lim_sup], color='orange', linestyle='--', linewidth=2, label='Diagonal ideal y = x')
    ax.set_title('RANSAC: observado vs proyectado', fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor observado u (px)')
    ax.set_ylabel('Valor proyectado u (px)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_ransac_summary(frames, inliers_counts, outliers_counts, mean_errors, max_errors):
    if len(frames) == 0:
        print('No hay frames válidos para resumir el RANSAC.')
        return

    frames = np.asarray(frames)
    mean_errors = np.asarray(mean_errors)
    modelo, mascara_inliers = ransac_linea_simple(frames, mean_errors, iteraciones=1000, umbral_residual=0.5)
    if modelo is None:
        print('No se pudo ajustar la regresión lineal con RANSAC.')
        return

    pendiente, intercepto = modelo
    outliers_mask = ~mascara_inliers
    x_linea = np.linspace(frames.min(), frames.max(), 200)
    y_linea = pendiente * x_linea + intercepto

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(frames[mascara_inliers], mean_errors[mascara_inliers], color='steelblue', s=65, edgecolors='black', linewidths=0.5, label=f'Inliers ({np.sum(mascara_inliers)})')
    if np.any(outliers_mask):
        ax.scatter(frames[outliers_mask], mean_errors[outliers_mask], color='red', marker='x', s=85, linewidths=2, label=f'Outliers ({np.sum(outliers_mask)})')
    ax.plot(x_linea, y_linea, color='orange', linewidth=2.5, label=f'Regresión RANSAC: y = {pendiente:.4f}x + {intercepto:.4f}')
    ax.set_title('Regresión lineal simple con RANSAC sobre todos los frames', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Error medio de reproyección (px)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
def plot_trajectory_and_markers(trayectoria_t, rotaciones=None):
    trayectoria_t = np.array(trayectoria_t)
    if len(trayectoria_t) == 0:
        print("No se calcularon traslaciones para graficar.")
        return
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    Mat_Rot_X = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    margen = 300
    x_min, x_max = 550 - margen, 2450 + margen
    y_min, y_max = 550 - margen, 1450 + margen
    esquinas_mesa = np.array([
        [x_min, y_min, -5],
        [x_max, y_min, -5],
        [x_max, y_max, -5],
        [x_min, y_max, -5]
    ])
    esquinas_mesa_rotadas = np.array([np.dot(Mat_Rot_X, pt) for pt in esquinas_mesa])
    verts = [list(zip(esquinas_mesa_rotadas[:, 0], esquinas_mesa_rotadas[:, 1], esquinas_mesa_rotadas[:, 2]))]
    mesa = Poly3DCollection(verts, alpha=0.2, facecolors='saddlebrown', edgecolors='black', linewidths=2)
    ax.add_collection3d(mesa)
    primero = True
    for aruco_id, pts in world_points_dict.items():
        puntos_rotados = np.array([np.dot(Mat_Rot_X, np.array([pt[0], pt[1], 0])) for pt in pts])
        x = np.append(puntos_rotados[:, 0], puntos_rotados[0, 0])
        y = np.append(puntos_rotados[:, 1], puntos_rotados[0, 1])
        z = np.append(puntos_rotados[:, 2], puntos_rotados[0, 2])
        if primero:
            ax.plot(x, y, z, color='gray', marker='o', linewidth=2, label='Marcadores ArUco')
            primero = False
        else:
            ax.plot(x, y, z, color='gray', marker='o', linewidth=2)
        ax.text(np.mean(x), np.mean(y), 0, f"ID {aruco_id}", color='black', fontsize=10, fontweight='bold')
    X = trayectoria_t[:, 0]
    Y = trayectoria_t[:, 1]
    Z = trayectoria_t[:, 2]
    if rotaciones is None:
        ax.plot(X, Y, Z, marker='o', linestyle='-', color='blue', label='Trayectoria', linewidth=1.5, markersize=3, alpha=0.7)
    else:
        ax.plot(X, Y, Z, linestyle='-', color='gray', label='Trayectoria', linewidth=1, alpha=0.5)
    if rotaciones is not None and len(rotaciones) == len(trayectoria_t):
        longitud_eje = 80.0
        paso_muestreo = max(1, len(rotaciones) // 20)
        for i in range(0, len(rotaciones), paso_muestreo):
            pos = trayectoria_t[i]
            R = rotaciones[i]
            ejes_camara = np.dot(Mat_Rot_X, R.T)
            eje_x = ejes_camara[:, 0]  
            eje_y = ejes_camara[:, 1]  
            eje_z = ejes_camara[:, 2]  
            ax.quiver(pos[0], pos[1], pos[2], 
                     eje_x[0] * longitud_eje, eje_x[1] * longitud_eje, eje_x[2] * longitud_eje,
                     color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(pos[0], pos[1], pos[2], 
                     eje_y[0] * longitud_eje, eje_y[1] * longitud_eje, eje_y[2] * longitud_eje,
                     color='green', arrow_length_ratio=0.1, linewidth=2)
            longitud_z = longitud_eje * 2.5
            ax.quiver(pos[0], pos[1], pos[2], 
                     eje_z[0] * longitud_z, eje_z[1] * longitud_z, eje_z[2] * longitud_z,
                     color='blue', arrow_length_ratio=0.15, linewidth=3)
            ax.scatter(pos[0], pos[1], pos[2], color='black', s=20)
    if rotaciones is not None and len(rotaciones) == len(trayectoria_t):
        import matplotlib.lines as mlines
        leyenda_x = mlines.Line2D([], [], color='red', label='Cámara: Eje X (Derecha)')
        leyenda_y = mlines.Line2D([], [], color='green', label='Cámara: Eje Y (Abajo)')
        leyenda_z = mlines.Line2D([], [], color='blue', label='Cámara: Eje Z (Frente/Lente)')
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([leyenda_x, leyenda_y, leyenda_z])
        ax.legend(handles=handles, loc='upper left')
        ax.set_title("Orientación y Trayectoria 3D (Ejes Locales de la Cámara)", fontsize=14, fontweight='bold')
    else:
        ax.legend(loc='upper left')
        ax.set_title("Trayectoria 3D de la Cámara (Sin Ejes)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_video = os.path.join(ruta_script, "VideoVision.mp4")
    archivo_parametros = os.path.join(ruta_script, "parametros_camara.npz")
    if not os.path.exists(archivo_parametros):
        print(f"Error: No se encontró {archivo_parametros}. Ejecuta calibrar_lente.py primero.")
        exit()
    datos_cargados = np.load(archivo_parametros)
    K_matrix = datos_cargados['K']
    print("\n" + "="*60)
    print("MATRIZ INTRÍNSECA K CARGADA:")
    print("="*60)
    print(K_matrix)
    print("="*60 + "\n")
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: No se pudo abrir el vídeo.")
        exit()
    diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parametros_aruco = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(diccionario_aruco, parametros_aruco)
    umbral_movimiento_mm = 10.0
    frames_minimo_espaciado = 2
    numero_frame = 0
    ultimo_frame_procesado = -frames_minimo_espaciado
    ultima_posicion_guardada = None
    trayectoria_t = []
    rotaciones = []
    errores_reproyeccion = []
    frames_ransac = []
    inliers_ransac = []
    outliers_ransac = []
    error_medio_ransac = []
    error_max_ransac = []
    print("--- PIPELINE PURISTA CARGADO (ALGEBRA LINEAL DIRECTA) ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esquinas, ids, _ = detector.detectMarkers(gray)
        marcadores_encontrados = {}
        if ids is not None:
            for i, aruco_id in enumerate(ids.flatten()):
                if aruco_id in ids_tablero and aruco_id not in marcadores_encontrados:
                    marcadores_encontrados[aruco_id] = esquinas[i][0]
                    cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)
        if len(marcadores_encontrados) == 4 and (numero_frame - ultimo_frame_procesado) >= frames_minimo_espaciado:
            ultimo_frame_procesado = numero_frame
            image_points = []
            world_points = []
            for aruco_id in ids_tablero:
                puntos_imagen = marcadores_encontrados[aruco_id]
                image_points.extend(puntos_imagen)
                world_points.extend(world_points_dict[aruco_id])
                centro_x = int(np.mean(puntos_imagen[:, 0]))
                centro_y = int(np.mean(puntos_imagen[:, 1]))
                cv2.circle(frame, (centro_x, centro_y), 10, (0, 255, 0), -1)
            world_pts_arr = np.array(world_points)
            img_pts_arr = np.array(image_points)
            H, inliers_mask, num_inliers = ransac_homografia(
                world_pts_arr, img_pts_arr, 
                iteraciones=1000, 
                umbral_error=3.0
            )
            errores_H, error_medio_H, error_max_H = calcular_error_homografia(world_pts_arr, img_pts_arr, H)
            frames_ransac.append(numero_frame)
            inliers_ransac.append(int(num_inliers))
            outliers_ransac.append(int(len(world_pts_arr) - num_inliers))
            error_medio_ransac.append(error_medio_H)
            error_max_ransac.append(error_max_H)
            # DESCOMPOSICIÓN DE LA HOMOGRAFÍA (EXTRACCIÓN DE R Y t)
            K_inv = np.linalg.inv(K_matrix)
            M = np.dot(K_inv, H)
            lambda_escala = 1.0 / np.linalg.norm(M[:, 0])
            r1 = M[:, 0] * lambda_escala
            r2 = M[:, 1] * lambda_escala
            r3 = np.cross(r1, r2)
            t = M[:, 2] * lambda_escala
            R_aprox = np.column_stack((r1, r2, r3))
            U_r, _, Vh_r = np.linalg.svd(R_aprox)
            R = np.dot(U_r, Vh_r)
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            if t[2] < 0:
                t = -t
                R[:, 0] *= -1
                R[:, 1] *= -1
            # TRANSFORMACIÓN AL SISTEMA DE REFERENCIA ROBÓTICO
            C_cv = -np.dot(R.T, t)
            Matriz_Rotacion_X_180 = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])
            C_robotica = np.dot(Matriz_Rotacion_X_180, C_cv)
            debe_guardar = False
            if ultima_posicion_guardada is None:
                debe_guardar = True
                distancia_mov = 0
            else:
                distancia_mov = np.linalg.norm(C_robotica - ultima_posicion_guardada)
                if distancia_mov >= umbral_movimiento_mm:
                    debe_guardar = True
            if debe_guardar:
                errores, error_medio, error_max, pts_proyectados = calcular_error_reproyeccion(
                    world_pts_arr, img_pts_arr, K_matrix, R, t
                )
                trayectoria_t.append(C_robotica)
                rotaciones.append(R)
                errores_reproyeccion.extend(errores)
                ultima_posicion_guardada = C_robotica.copy()
                print(f"✓ Frame {numero_frame} GUARDADO (movimiento: {distancia_mov:.2f}mm)")
                cv2.putText(frame, f"POSE GUARDADA (Frame {numero_frame})", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            else:
                cv2.putText(frame, f"Detectado pero no guardado (Frame {numero_frame})", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
            alto, ancho = frame.shape[:2]
            escala = 1000.0 / max(alto, ancho)
            frame_mostrar = cv2.resize(frame, (int(ancho * escala), int(alto * escala))) if escala < 1.0 else frame
            cv2.imshow('Recoleccion Pure-Math', frame_mostrar)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            alto, ancho = frame.shape[:2]
            escala = 1000.0 / max(alto, ancho)
            frame_mostrar = cv2.resize(frame, (int(ancho * escala), int(alto * escala))) if escala < 1.0 else frame
            cv2.imshow('Recoleccion Pure-Math', frame_mostrar)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        numero_frame += 1
    cap.release()
    cv2.destroyAllWindows()
    print("\nVídeo finalizado. Generando gráficas...")
    plot_ransac_summary(frames_ransac, inliers_ransac, outliers_ransac, error_medio_ransac, error_max_ransac)
    plot_trajectory_and_markers(trayectoria_t, rotaciones)
    if len(errores_reproyeccion) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(errores_reproyeccion, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel('Error de Reproyección (píxeles)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Errores de Reproyección')
        plt.grid(True, alpha=0.3)
        plt.axvline(np.mean(errores_reproyeccion), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(errores_reproyeccion):.3f}px')
        plt.axvline(np.median(errores_reproyeccion), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(errores_reproyeccion):.3f}px')
        plt.legend()
        plt.tight_layout()
        plt.show()
