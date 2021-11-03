## Dogs-vs-Cats dataset
https://www.kaggle.com/c/dogs-vs-cats
- [Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- Download and extract the dataset such that train.zip in root directory

## Setup
```shell
pip install -r requirements.txt
```

## Training
```shell
python train.py --gpus 1 --max_epochs 10
```
- Example output: 
```
best checkpoint lightning_logs/ckpt1/dog-cat-resnet18-epoch=10-val_loss=0.09.ckpt
```

## Inference
- Download an image of dog or cat
```shell
python inference.py --model lightning_logs/ckpt1/dog-cat-resnet18-epoch\=10-val_loss\=0.09.ckpt --image dog1.jpg
```
- Example output:
```shell
{'class_idx': 1, 'class_name': 'dog'} duration 10.766785383224487 sec(s) per 1000 predictions
```

## API
- Start the server:
```shell
uvicorn main:app --reload
```
- Go to [localhost:8000/docs](http://localhost:8000/docs) and try to upload an image of dog or cat at /predict rout
