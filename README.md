
# SuNBEaM

This package provides code to compute the (S)pectral (N)on-(B)acktracking
(E)igenvalue Pseudo-(M)etric, or SuNBEaM for short.  The non-backtracking
matrix is a matrix representation of a graph that has deep connections with
the theory of the length spectrum.  The eigenvalues of the non-backtracking
matrix can be effectively used to compute dissimilarity scores (or
distances) between graphs.  For more information, see

> Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad, *Graph Distance from
> the Topological View of Non-Backtracking Cycles*.

All experiments and figures in this paper were generated with an earlier
version of the code in this repository.

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

er = nx.erdos_renyi_graph(300, 0.05)
ba = nx.barabasi_albert_graph(300, 3)
sunbeam.nbdist(er, ba, 20)
# <- 13.938343758008838
```

A more extensive example of the functionality provided in `sunbeam.py` can
be found in the [example notebook](https://github.com/leotrs/sunbeam/blob/master/example.ipynb).
