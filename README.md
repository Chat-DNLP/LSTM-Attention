# Implementación de un mecanismo de atención en un modelo Seq2Seq con LSTMs

<div align="justify">
  
Este repositorio contiene la implementación de un modelo *Seq2Seq* Long Short-Term Memory (LSTM) y un mecanismo de atención, siguiendo los enfoques de Bahdanau y Luong. La implementación de estos mecanismos de atención mejora la capacidad del modelo para focalizarse en las partes relevantes de las secuencias de entrada, facilitando una generación de secuencias de salida más precisa y contextualizada.

## 1. Contenido

- `/attention`: Este directorio contiene la implementación del mecanismo de atención, estructurada en clases y organizada para soportar diversas operaciones de atención, tales como producto punto a punto, atención bilineal (también conocida como "atención de Luong") y atención mediante perceptrón multicapa (conocida como "atención de Bahdanau"). La Figura 1 presenta el diagrama de clases que ilustra la estructura de este módulo.

<div align="center">
  <img src="images_readme/attention_uml.png" alt="Diagrama UML atención" />
    <p><strong>Figura 1.</strong> Diagrama de clases del módulo <i>Attention</i>.</p> 
</div>

- `/models_definition`:  Este directorio alberga la definición de los diferentes modelos: Seq2Seq con atención, incluyendo las variantes de Luong y Bahdanau. La Figura 2 presenta un diagrama de clases que proporciona una síntesis básica de estos modelos; este diagrama no contiene especificaciones detalladas.

<div align="center">
  <img src="images_readme/models_uml.png" alt="Diagrama UML modelos" width =700 />
    <p><strong>Figura 2.</strong> Diagrama de clases de la definición de los modelos.</p> 
</div>

- `LSTM-Notebook.ipynb`: Este *notebook* implementa un modelo de traducción automática neuronal utilizando arquitecturas *Seq2Seq* con LSTM y mecanismos de atención (Bahdanau o Luong). Su objetivo es explorar y entender el funcionamiento de estos modelos mediante la carga de datos, el entrenamiento y la evaluación, además de integrar la herramienta *Weights & Biases* para el seguimiento de experimentos.
- `sanity_check.ipynb`: Este *notebook* realiza un *sanity check* del mecanismo de atención en un modelo de traducción, comprobando la forma y el cálculo de los pesos de atención para asegurar la correcta implementación de las operaciones en PyTorch.
- `translation.py`: Este archivo define una clase de dataset personalizado para tareas de traducción entre inglés y español en PyTorch, la cual carga, tokeniza, vectoriza y separa los datos de texto en conjuntos de entrenamiento y prueba, incorporando tokens especiales (`<sos>`, `<eos>`, `<pad>`, `<unk>`) y adaptando el tamaño de secuencias mediante padding para facilitar el procesamiento en los modelos.


</div>
