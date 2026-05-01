# Sistema de Visión 3D - Eurobot

Pipeline purista basado en Álgebra Lineal para la estimación de trayectorias y pose de cámara (6DoF). Este sistema implementa el método matemático de Zhang para la calibración intrínseca y utiliza descomposición de homografía robusta mediante RANSAC para el rastreo de marcadores ArUco en tiempo real, permitiendo una reconstrucción precisa de la trayectoria de la cámara sin depender de librerías de alto nivel.

## Librerías Necesarias
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`

## Ejecución del Proyecto
Para ejecutar todo el sistema automáticamente, utiliza el siguiente comando en la terminal:

```bash
python3 main.py
```
*(Nota: Durante la ejecución se abrirá una gráfica de calibración. Ciérrala para que el programa pueda continuar con la simulación 3D).*

## Descripción de los Scripts
- **`main.py`**: Orquestador principal. Ejecuta el pipeline completo de manera secuencial.
- **`calibrar_lente.py`**: Aplica el método matemático de Zhang para calcular la matriz intrínseca de la cámara a partir del vídeo y la guarda en un archivo.
- **`calibracion.py`**: Carga la matriz de la cámara y rastrea los ArUcos del vídeo para estimar y visualizar la trayectoria 3D de la cámara en el espacio.
