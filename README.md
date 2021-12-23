# SR-HyperFace
KAIST 2020 Fall Semester - CS495: Individual Study

KAIST GCLab Yeonghun Kim, Sanghyun Jung

-------------------
Improvement of HyperFace using Super Resolution

## Usage
### Environment
* Windows 10
* Python 3.6
* CUDA 10.1

### Perparation
* AFLW dataset in ```dataset/aflw/```
* Pretrained ESRGAN model in ```ESRGAN/checkpoints/```

### Training
* Train HyperFace on AFLW Dataset
```
python train_hyperface.py
```

* Train HyperFace on SR-AFLW Dataset
```
python train_srhyperface.py
```

## References
* Ranjan, Rajeev, Vishal M. Patel, and Rama Chellappa. **"Hyperface: A deep multi-task learning framework for face detection, landmark localization, pose estimation, and gender recognition."** IEEE transactions on pattern analysis and machine intelligence 41.1 (2017): 121-135.
* Wang, Xintao, et al. **"Esrgan: Enhanced super-resolution generative adversarial networks."** Proceedings of the European conference on computer vision (ECCV) workshops. 2018.
