
# SuNBEAM

This package provides code to compute the (S)pectral (N)on-(B)acktracking
(E)mbedding (A)nd Pseudo-(M)etric, or SuNBEAM for short.  The
non-backtracking matrix is a matrix representation of a graph that has deep
connections with the theory homotopy of graphs, in particular the length
spectrum function.  The eigenvalues of the non-backtracking matrix can be
effectively used to compute dissimilarity scores (or distances) between
graphs.  An old version of our manuscript can be found at the following
link.  (Newer version currently under review.)

> Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad, **Graph Distance from
> the Topological View of Non-Backtracking Cycles**. Preprint,
> arXiv:1807.09592 [cs.SI], (2018).

All experiments and figures in this paper were generated with an earlier
version of the code in this repository.

<p align="center">
  <img src="https://github.com/leotrs/sunbeam/blob/master/random_eigenvalues.png?raw=true" alt="random eigenvalues"/>
</p>


# Installation

To install, simply `git clone` this repository, import the `sunbeam` module
and call the `nbd` function.  For `sunbeam` to work correctly you need to
have installed NumPy, SciPy, NetworkX, POT, and (optionally) Matplotlib.
To install [POT](https://pot.readthedocs.io/en/stable/), run
```
$ pip install POT==0.5.1
```

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
