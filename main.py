import subprocess
import os
import sys

def main():
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    script_calibrar_lente = os.path.join(ruta_script, 'calibrar_lente.py')
    script_calibracion = os.path.join(ruta_script, 'calibracion.py')
    print('=' * 60)
    print('INICIANDO PIPELINE DE VISIÓN 3D - EUROBOT')
    print('=' * 60)
    print('\n[PASO 1] Ejecutando calibración de la cámara (Fase 1-5 Zhang)...')
    print("Esto calculará los parámetros intrínsecos y los guardará en 'parametros_camara.npz'.")
    print('(Nota: Cierra la gráfica generada para que el pipeline continúe al Paso 2)')
    try:
        subprocess.run([sys.executable, script_calibrar_lente], check=True)
    except subprocess.CalledProcessError:
        print('\n[ERROR] Fallo en la calibración de la cámara. Deteniendo el pipeline.')
        sys.exit(1)
    print('\n[PASO 1 COMPLETADO] Parámetros de cámara generados correctamente.')
    print('\n[PASO 2] Ejecutando estimación de trayectoria 3D y visualización...')
    print('Esto usará los parámetros de la cámara para rastrear los ArUcos y mostrar la gráfica 3D.')
    try:
        subprocess.run([sys.executable, script_calibracion], check=True)
    except subprocess.CalledProcessError:
        print('\n[ERROR] Fallo en la estimación de trayectoria.')
        sys.exit(1)
    print('\n' + '=' * 60)
    print('PIPELINE FINALIZADO CON ÉXITO')
    print('=' * 60)
if __name__ == '__main__':
    main()