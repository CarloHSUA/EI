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
- `extended_model_run_level_GHQ.pth`
- `extended_model_run_level_GHQ.pth`
- `extended_model_run_level_GHQ.pth`
- `extended_model_run_level_GHQ.pth`


Ejecutando el siguiente comando:
```
python main.py <nombre_del_fihero_de_pesos>
```

## Estructura del repositorio
- data/ : Contiene el dataset de S-FFSD en diferentes archivos .cvs.
- img/ : Contiene las imagenes de los resultados generados en el programa prediction.py. 
- methods/ : Contiene las implementaciones de los modelos, en este caso, solo el GTAN.
- models/ : Contiene el modelo GTAN preentrenado. Su extensión es .pth.
- new_model/ : Contiene el modelo GTAN entrenado y alamacenado tras ejecutar testing.py. Su extensión es .pth.

## Referencias
Gran parte del código presente esta extraido del repositorio [GaitGraph2](https://github.com/tteepe/GaitGraph2)
