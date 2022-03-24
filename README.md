## FaceForencics Detection
~~Use `FaceForencics++` dataset to train model, get total 0.62 in benchmark now.~~

## Model
Use Resnext50 to train `NeuralTextures`, `Deepfakes` and `Original` data, use some data augmentation process.

## Train
### Environment
```shell
# install pytorch with corresponding cuda version
pip install torch

# install facenet-pytorch
pip install facenet-pytorch
```

### Get Train Data
Get `FaceForencics++` dataset, put them to `dataset` folder. My folder structure is like following.
```
--- dataset
    --- faceforensics_benchmark_images
    --- mtcnn
        --- deepfakes_faces_c23
        --- deepfakes_faces_c40
        --- face2face_faces_c23
        --- face2face_faces_c40
        --- faceswap_faces_c23
        --- faceswap_faces_c40
        --- faceswap_faces_raw
        --- neural_textures_faces_c23
        --- neural_textures_faces_c40
        --- neural_textures_faces_raw
        --- original_faces_c23
        --- original_faces_c40
        --- original_faces_raw
    --- retina
        --- deepfakes_faces_c23
        --- deepfakes_faces_c40
        --- face2face_faces_c23
        ...
    c40_all.pkl
    data_all.pkl
    retina_all.pkl
```

Use script to get images' path pkl.
```shell
cd src
python get_train_val_data.py
```

Run `train_resnext.py` script.
```shell
cd src
python train_resnext.py
```