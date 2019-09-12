# PiNNboard

PiNNboard is a plugin for Tensorboard to visualize atomic neural
networks. It provides a way to define and visualize the activate
and weights of atomic neural networks, with special focus on
the PiNet neural network.

This visualization is much inspired by the 
[tensorboard playground](https://playground.tensorflow.org/)
in terms of both the idea and the style.

![](demo.gif)

# Installation
PiNNboard currently depends on the nightly version of tensorboard for
dynamical plugin support. To install:

```bash
# Preferably do this in a virtual env.
pip install tf-nightly
# clone the repository
cd PiNNboard && pip install -e .
```

# Usage

For visualizing a PiNet, see the example in the `notebooks/` folder.

# Known issues

**WARNING** PiNNboard is currently under active development, and it
relies on experimental features of Tensorboard.

- The dynamic plugin feature is only available in the nightly
  verion of tensorboard, which is expected to move to TF 2.0
  soon.
- The fontend part of the code has much to improve, e.g. more options,
  better UI, etc.
