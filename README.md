Integrantes del grupo: Silvana Araya, Valentina Martinez, Sayen Barra

CLASIFICACION DE EMOCIONES EN VIDEOJUEGOS CON RESNET50V2
 
DESCRIPCION DEL PROYECTO
   
  Este proyecto consiste en un clasificador de imagenes que reconoce 5 emociones
  (anger, fear, happiness, sadness, serenity) a partir de capturas de videojuegos.
  Se uso un dataset propio de 1000 imagenes (200 por cada emocion).
   
  Para el entrenamiento se aplico transfer learning usando ResNet50V2 preentrenada
  en ImageNet, agregando capas densas propias para clasificar en las 5 categorias.
  Se entrenaron tres versiones del modelo:
   
  - v1: ResNet50V2 base + capas densas (accuracy 53.5%)
  - v2: igual que v1 pero con augmentation avanzado (CutMix, MixUp, CutOut, Color
    Jitter, Random Resized Crop) (accuracy 55.0%, fue el mejor modelo)
  - v3: fine-tuning de las ultimas 30 capas de ResNet + class weights + label
    smoothing (accuracy 54.5%)
   
  Tambien se hizo un analisis extra para ver si el modelo depende del color para
  decidir, comparando predicciones sobre imagenes originales vs en blanco y negro.
   
  Para la parte de inferencia, se armo una funcion que dibuja la prediccion
  (emocion + porcentaje de confianza) directamente sobre la imagen usando OpenCV,
  y se probo el modelo con imagenes nuevas que no son parte del dataset original,
  para ver que tan bien generaliza.
   
   
ESTRUCTURA DEL DATASET
   
  games_mood/
    anger/       (200 imagenes)
    fear/        (200 imagenes)
    happiness/   (200 imagenes)
    sadness/     (200 imagenes)
    serenity/    (200 imagenes)
   
  Se dividio en 80% entrenamiento (800 imagenes) y 20% validacion (200 imagenes).
   
   
LIBRERIAS NECESARIAS
   
  Para correr el codigo se necesitan estas librerias de Python:
   
  pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pillow
    
   - tensorflow / keras: para armar, entrenar y cargar los modelos
   - opencv-python (cv2): para dibujar la prediccion sobre la imagen y pasar a
     blanco y negro en el experimento de color
   - numpy: manejo de imagenes como arrays
   - matplotlib: graficos y visualizaciones
   - seaborn: matrices de confusion
   - scikit-learn: metricas (matriz de confusion, classification report, class
     weights)
   - Pillow (PIL): abrir y manipular imagenes
   
  El proyecto se entreno en Google Colab usando GPU. Si solo se quiere hacer
  inferencia con un modelo ya entrenado no hace falta GPU, se puede correr en
  cualquier computador con CPU.
   
   
COMO EJECUTAR EL PROYECTO
   
  1. Clonar el repositorio y tener la carpeta games_mood con el dataset.
  2. Instalar las librerias mencionadas arriba.
  3. Abrir el notebook .ipynb en Google Colab o en Jupyter/VS Code.
  4. Cambiar las rutas (drive_path, dataset_path, rutas de los .h5) segun donde
     esten los archivos.
  5. Si solo se quiere probar el modelo ya entrenado, no hace falta reentrenar,
     basta con cargarlo asi:
   
     from tensorflow.keras.models import load_model
     model_new = load_model('modelo_v2_augmentation.h5')
   
  6. Correr las funciones de prediccion (predict_emotion_final,
     dibujar_prediccion_cv2) sobre las imagenes de prueba.
   
PIPELINE DE INFERENCIA
 
La funcion dibujar_prediccion_cv2() escribe la emocion predicha y el porcentaje
de confianza directamente sobre los pixeles de la imagen, usando solo OpenCV:
 
def dibujar_prediccion_cv2(img_path, pred, conf):
    
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    texto = f"{pred.upper()} ({conf:.1%})"
    cv2.rectangle(img, (10, 10), (400, 60), (0, 0, 0), -1)
    cv2.putText(img, texto, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

    
EJEMPLO DE INFERENCIA
   
Ver imagen: example_inference.png
(prediccion del modelo v2 sobre una imagen nueva, que no es parte del dataset)
   
   
EVALUACION CON DATOS NUEVOS
   
  Se probo el modelo con imagenes externas al dataset, que no se usaron ni en
  entrenamiento ni en validacion. El analisis de estos resultados, junto con la
  comparacion entre v1, v2 y v3 y el experimento del color, esta en celdas de
  texto dentro del notebook.
   
  Principales conclusiones:
      - El modelo v2 (con augmentation avanzado) fue el que mejor resultado dio en
        general (55.0% de accuracy).
      - La emocion serenity es la que el modelo reconoce mejor en los tres modelos.
      - La emocion sadness es la mas dificil, se confunde bastante con happiness.
      - El modelo parece fijarse mas en las formas y estructuras de la imagen que en
        el color para decidir la emocion.
