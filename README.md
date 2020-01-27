

# SuNBEAM

This package provides code to compute the (S)pectral (N)on-(B)acktracking
(E)mbedding (A)nd Pseudo-(M)etric, or SuNBEAM for short.  The
non-backtracking matrix is a matrix representation of a graph that has deep
connections with the theory homotopy of graphs, in particular the length
spectrum function.  The eigenvalues of the non-backtracking matrix can be
effectively used to compute dissimilarity scores (or distances) between
graphs.  A previous version of the code in this repository was used in the
following paper.

> Leo Torres, Pablo Suárez-Serrato, and Tina Eliassi-Rad:
> **Non-backtracking cycles: length spectrum theory and graph mining
> applications**. Applied Network Science 4(1): 41:1-41:35 (2019)

<p align="center">
  <img src="https://github.com/leotrs/sunbeam/blob/master/random_eigenvalues.png?raw=true" alt="random eigenvalues"/>
</p>


# Installation

To install, simply `git clone` this repository, import the `sunbeam` module
and call the `nbd` function.  For `sunbeam` to work correctly you need to
have installed NumPy, SciPy, NetworkX, and (optionally) Matplotlib.


# Example

A minimal example of how to use `sunbeam.py`:

```
import sunbeam
import networkx as nx

# Compute distances NBD
er = nx.erdos_renyi_graph(300, 0.05)
ba = nx.barabasi_albert_graph(300, 3)
sunbeam.nbd(er, ba, 20)                      # 1.6322743723800643

# Compute embedding NBED
2*er.size(), sunbeam.nbed(er).shape[0]
(4580, 4580)

# Visualize NBED
sunbeam.visualize_nbed(er, color='target')
sunbeam.visualize_nbed(ba, color='source', log=True)
```

A more extensive example of the functionality provided in `sunbeam.py` can
be found in the [example notebook](https://github.com/leotrs/sunbeam/blob/master/example.ipynb).


# Citation

When using code in this repository, please cite both the paper and the
repository,

```
@article{TorresSE19,
  author    = {Leo Torres and
               Pablo Su{\'{a}}rez{-}Serrato and
               Tina Eliassi{-}Rad},
  title     = {Non-backtracking cycles: length spectrum theory and graph mining applications},
  journal   = {Applied Network Science},
  volume    = {4},
  number    = {1},
  pages     = {41:1--41:35},
  year      = {2019}
}

@misc{TorresSuNBEaM,
  author    = {Leo Torres},
  title     = {{SuNBEaM}: Spectral Non-Backtracking Embedding And pseudo-Metric},
  year      = {2018},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/leotrs/sunbeam}},
}

```
