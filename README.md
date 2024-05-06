# EI

## Requisitos

1. Instalar las dependencias necesarias.
```
pip install -r requirements.txt
```

2. Descomprimir el dataset
```
cd psimo_reduced
unzip semantic_data_new.zip
```

3. Crear JSON con los datos del esqueleto
```
python preprocess_psimo.py 
```

## Ejecución
Una vez instaladas todas las dependencias necesarias, podemos ejecutar el siguiente programa. Dicho programa hará un entrenamiento del modelo ResGCN-V2 si no encuentra el fichero con los pesos en la carpeta de `model/extended_model.pth`.

```
python main.py
```

En el caso de que se pretenda usar un modelo preentrenado, se pueden utilizar los siguientes:
- `extended_model_run_level_GHQ.pth`
- `extended_model_run_level_RSE.pth`
- `extended_model_subject_level_GHQ.pth`
- `extended_model_subject_level_RSE.pth`
- `basic.ckpt`


Ejecutando el siguiente comando:
```
python main.py <nombre_del_fihero_de_pesos>
```

## Estructura del repositorio
- data/ : Tras ejecutar `preprocess_psimo.py` se generan JSON y almacenan en dicha carpeta.
- model/ : Contiene los .pth.
- psimo_reduced/ : Contiene los datos del dataset comprimidos en .zip.
- ResGCNv1/ : Contiene el código de la arquitectura ResGCN.
- st_gcn/ : Contiene el código de la arquitectura ST-GCN.
- transforms/ : Transformaciones que se pueden aplicar al conjunto de datos.

## Referencias
Gran parte del código presente esta extraido del repositorio [GaitGraph2](https://github.com/tteepe/GaitGraph2)
