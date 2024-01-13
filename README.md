# baby_mamba
An implementation of Mamba to develop an understanding of its functioning.

### Intro:
Mamba is a recent sequence modelling architecture proposed by Albert Gu and Tri Dao (https://arxiv.org/abs/2312.00752). Mamba selling points are fast inference (5x faster than transformers) and linear scaling with sequence length. 

Mamba main building block is selective state spaces, which is a computation of the following equations:


$$
h(t + 1) = Ax(t) + Bu(t)
$$

$$
y(t) = Ch(t) + Du(t)
$$

For discret modalities like texts and genomics, A,B will be discretised by a timestep ∆.

**Interpretation of ∆:**
- *Function:* ∆ controls the balance between focusing on or ignoring the current input \(u_t\).
- *Effect:* A large ∆ resets the state \(h\) and emphasizes the current input, while a small ∆ maintains the state, effectively ignoring the current input.
- *Relation to SSMs:* Equations (1)-(2) can be seen as a continuous system discretized by a timestep ∆. A large ∆ represents the system focusing on the current input for an extended duration, while a small ∆ signifies a transient input that is ignored.

**Interpretation of A:**
- *Role:* A parameter interacts with ∆ through \(A = \exp(\Delta A)\) (discretisation equation).
- *Effect:* The selectivity in ∆ is crucial to ensure selectivity in (A, B), and it serves as the primary source of improvement.
- *hypothesis (Mamba paper page 9):* Making A selective in addition to or instead of ∆ could yield similar performance, but for simplicity, it is left out.

**Interpretation of B and C:**
- *Importance:* Selectivity is vital for filtering out irrelevant information, allowing a sequence model to compress context efficiently.
- *Role of B and C:* Modifying B and C to be selective provides finer control over incorporating input \(u_t\) into the state \(h_t\) or the state into the output \(y_t\).
- *Interpretation:* B and C enable the model to modulate recurrent dynamics based on content (input) and context (hidden states), respectively.

This computation (i.e SSM block) is implemented in *selective_scan.py* (it's the reference implementation from [mamba code base](https://github.com/state-spaces/mamba/), and it's not the hardware-aware implementation, check: https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops for more info)

The Figure below describes the interaction of these equations:

![SSM Block](./assets/ssm.png)

Just like other mainstream architectures, Mamba interleaves this computation (SSM block) with MLP layers, a normalisation technique and a residual connections, they have also added a conv layer (Figure below):

<p align="center">
    <img alt="Mamba Architecture" src="./assets/mamba_architecture.png">
</p>


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

