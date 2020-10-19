# Exponential ergodicity of mirror-Langevin diffusions

This repository is the official implementation of [Exponential ergodicity of mirror-Langevin diffusions](https://arxiv.org/abs/2005.09669), and in particular for the Newton Langevin Algorithm. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Sampling 

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation 

To evaluate samples,

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Results



