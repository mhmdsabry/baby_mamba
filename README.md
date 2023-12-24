# baby_mamba
An implementation of Mamba to develop an understanding of its functioning.

### Intro:
Mamba is a recent sequence modelling architecture proposed by Albert Gu and Tri Dao (https://arxiv.org/abs/2312.00752). Mamba selling points are fast inference (5x faster than transformers) and linear scaling with sequence length. 

Mamba main building block is selective state spaces, which is a computation of the following equations:

```

```

This computation (i.e SSM block) is implemented in *selective_scan.py* (it's the reference implementation from [mamba code base](https://github.com/state-spaces/mamba/), and it's not the hardware-aware implementation, check: https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops for more info)

The Figure below describe the interaction of these equations:

Just like other mainstream architecture, Mamba interleaves this computation (SSM block) with MLP, normalisation technique and a residual connections, they also added a conv layer (Figure below):

### run code:

##### Environment:
```
conda env create -f environment.yml
conda activate mamba_env
```
##### Train a Mamba
config.ini contains model, and training arguments, feel free to play, and run the following to train:
```
python main.py -c config.ini
```

##### Eval loss:


##### Generated text:
```
```

