# Tesis: Modelo Ensamblado

Este proyecto presenta un modelo ensamblado que utiliza tres modelos base para realizar predicciones. El código está diseñado para ser ejecutado en un entorno de Python 3.8 y requiere un conjunto de dependencias especificadas en el archivo `requirements.txt`. A continuación, se detallan los pasos para instalar las dependencias, configurar el entorno, descargar el dataset y ejecutar el proyecto.

## Requisitos

- **Python**: Se recomienda utilizar Python 3.8 para garantizar la compatibilidad con las dependencias y el código del proyecto.
- **Dependencias**: El proyecto utiliza varias bibliotecas que están especificadas en el archivo `requirements.txt`. Asegúrate de instalar todas las dependencias necesarias antes de ejecutar el código.

## Pasos de Instalación

1. **Instalar Python 3.8** (si aún no lo tienes instalado):

   - Puedes descargar la versión 3.8 de Python desde el sitio oficial: [https://www.python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/).

2. **Clonar el Repositorio**:

   Si aún no tienes el proyecto en tu máquina local, clónalo utilizando `git`:

   ```bash
   git clone https://github.com/ehuallap/ensembled-model-thesis
   cd ensembled-model-thesis
   ```

3. **Crear y Activar un Entorno Virtual** (opcional pero recomendado):

   Es recomendable crear un entorno virtual para evitar conflictos de dependencias con otros proyectos.

   - Crear el entorno virtual:

     ```bash
     python3.8 -m venv venv
     ```

   - Activar el entorno virtual:

     - En Windows:

       ```bash
       .\venv\Scripts\activate
       ```

     - En macOS/Linux:

       ```bash
       source venv/bin/activate
       ```

4. **Instalar las Dependencias**:

   Una vez que el entorno virtual esté activo, instala todas las dependencias necesarias utilizando el archivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Descargar el Dataset**:

   El proyecto requiere un dataset para el entrenamiento y las pruebas. Puedes descargarlo desde el siguiente enlace:

   ```text
   https://drive.google.com/file/d/1NedGwmIIXFRL3YyUMygS35jgKu2vxTA9/view?usp=sharing
   ```

   Una vez descargado, extrae el contenido del archivo y colócalo dentro de la carpeta `data/merged_dataset` en el directorio del proyecto.

   La estructura de directorios debe ser la siguiente:

   ```text
   └── ensembled-model-thesis
       └── data/
           └── merged_dataset/
               └── <contenido_del_dataset>
   ```

6. **Ejecutar el Proyecto**:

   - **Entrenamiento**: Para entrenar el modelo ensamblado, ejecuta el siguiente comando:

     ```bash
     python ensembled.py
     ```

   - **Pruebas**: Para realizar las pruebas y evaluaciones con el modelo entrenado, ejecuta el siguiente comando:

     ```bash
     python test.py
     ```

## Estructura del Proyecto

El proyecto se organiza de la siguiente manera:

```text
<nombre_del_directorio_del_proyecto>/
│
├── ensembled.py            # Script principal para el entrenamiento del modelo ensamblado
├── test.py                 # Script para realizar pruebas y evaluaciones del modelo
├── requirements.txt        # Archivo con las dependencias del proyecto
├── data/                   # Carpeta que contiene los datasets
│   └── merged_dataset/     # Subcarpeta con el dataset descargado
└── README.md               # Este archivo
```

## Notas Adicionales

- Si encuentras algún problema con las dependencias o el entorno, asegúrate de estar utilizando Python 3.8 y de que todas las bibliotecas estén correctamente instaladas.
- Para cualquier consulta o sugerencia, no dudes en abrir un "issue" en el repositorio.