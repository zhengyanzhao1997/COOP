

![Image text](https://github.com/zhengyanzhao1997/COOP/blob/master/imgs/headpic.png)

Este repositorio es la implementación oficial del artículo ICCV2023 "COOP: Desacoplamiento y acoplamiento de la generación de poses de agarre de cuerpo completo".

COOP es un marco novedoso para sintetizar poses de cuerpo completo realistas que cubren el rango más amplio de capacidades de agarre humano.
![Image text](https://github.com/zhengyanzhao1997/COOP/blob/master/imgs/coop.jpg)

Para más detalles, consulte el artículo <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_COOP_Decoupling_and_Coupling_of_Whole-Body_Grasping_Pose_Generation_ICCV_2023_paper.html" target="_blank">COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation</a>.

## Instalación

Para instalar las dependencias, siga los siguientes pasos:

- Clone este repositorio: 
    ```Shell
    git clone https://github.com/zhengyanzhao1997/COOP
    cd COOP
    ```
- Instale las dependencias con el siguiente comando:
    ```
    pip install -r requirements.txt
    ```

## Primeros pasos

#### BNet y HNet
- Hemos liberado los checkpoints de BNet y HNet en las carpetas que se muestran a continuación.
```bash
    COOP
    ├── src
    │   ├── models
    │       ├── Bnet
    │       └── Hnet
```

#### Descargar modelos SMPLX y MANO
- Descargue los modelos corporales siguiendo los pasos en el [repositorio de SMPLX](https://github.com/vchoutas/smplx).
- Descargue los modelos corporales siguiendo los pasos en el [repositorio de MANO](https://mano.is.tue.mpg.de/).

- Coloque los modelos SMPLX y MANO en las carpetas que se muestran a continuación.
```bash
    COOP
    ├── model
    │   ├── smplx
    │   └── mano
```

### Generar agarres de cuerpo completo para el conjunto de prueba.

    cd src

Luego

    python infer/infer.py --rh-work-dir ../model/Hnet/ --body-work-dir../model/Bnet/  --data-file ../data --save-file [your save path]
    
También puede especificar el nombre del objeto de prueba, la forma del sujeto y la posición del objeto de prueba:

    python ... --object apple --subject s1 --x 0.4 --z 0.4 --y 0.4

El resultado visualizado (.html) y el archivo de malla generado se guardarán en su ruta de guardado.

### Entrenar su propio COOP.
próximamente

## Cita
```
@InProceedings{Zheng_2023_ICCV,
    author    = {Zheng, Yanzhao and Shi, Yunzhou and Cui, Yuhao and Zhao, Zhongzhou and Luo, Zhiling and Zhou, Wei},
    title     = {COOP: Decoupling and Coupling of Whole-Body Grasping Pose Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {2163-2173}
}
```
