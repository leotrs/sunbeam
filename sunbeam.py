"""
sunbeam.py
-----------

(S)pectral (N)on-(B)acktracking (E)igenvalue Pseudo-(M)etric. This module
contains three kinds of functions related to the computation of distances
between graphs using the eigenvalues of the non-backtracking matrix.

## Matrix Representations

The function `fast_hashimoto` computes the Hashimoto or `non-backtracking`
matrix of a graph, while `pseudo_hashimoto` computes a block matrix whose
eigenvalues are also eigenvalues of the Hashimoto matrix.  The function
`half_incidence` returns the half-incidence matrices of the graph.  These
are useful when computing the Hashimoto matrix or its eigenvalues
efficiently.  See discussion in [1].  Building the pseudo-Hashimoto matrix
is always considerably faster than building the Hashimoto matrix.  Building
the former takes as long as building the adjacency matrix of the graph.
Building the latter scales as the second moment of the degree distribution
of the graph (i.e., graphs with heavy-tail degree distributions will take
longer).

## Eigenvalue function

The function `nbeigs` returns the non-backtracking eigenvalues of a graph.
The eigenvalues may be returned as complex numbers, as a 2D array, or as a
1D 'unrolled' feature vector.  Different formats are convenient for
different purposes.  In computing the eigenvalues, we first obtain the
2-core of the graph by a process known as 'shaving'.  See algorithm 5.1 of
[1].

## Distance function

The function `nbdist` takes two graphs and returns the distance between
their non-backtracking eigenvalue spectra.  It also offers the optional
fine-tuning parameters sigma and eta.  These are used to make `nbdist` more
versatile and able to emphasize triangles or degree distribution.  For more
information, see [1].

## References

[1] Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad, **Graph Distance
    from the Topological View of Non-Backtracking Cycles**, preprint,
    arXiv:1807.09592 [cs.SI], (2018).

Autor: leotrs (leo@leotrs.com)

"""

import numpy as np
from numpy.linalg import norm
import networkx as nx
import scipy.sparse as sparse

TOL = 1e-5


#############################
# 1. Matrix representations #
#############################

def pseudo_hashimoto(graph):
    """Return the pseudo-Hashimoto matrix.

    The pseudo Hashimoto matrix of a graph is the block matrix defined as
    B' = [0  D-I]
         [-I  A ]

    Where D is the degree-diagonal matrix, I is the identity matrix and A
    is the adjacency matrix.  See [1] for the relationship between the
    Hashimoto matrix B and the pseudo-Hashimoto matrix B'.

    [1] Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad
        Graph Distance from the Topological View of Non-Backtracking Cycles.

    Params
    ------

    graph (nx.Graph): A NetworkX graph object.

    Returns
    -------

    A sparse matrix in csr format.

    """
    # Note: the rows of nx.adjacency_matrix(graph) are in the same order as
    # the list returned by graph.nodes().
    degrees = graph.degree()
    degrees = sparse.diags([degrees[n] for n in graph.nodes()])
    adj = nx.adjacency_matrix(graph)
    ident = sparse.eye(graph.order())
    pseudo = sparse.bmat([[None, degrees - ident], [-ident, adj]])
    return pseudo.asformat('csr')


def half_incidence(graph, ordering='blocks', return_ordering=False):
    """Return the 'half-incidence' matrices of the graph.

    If the graph has n nodes and m *undirected* edges, then the
    half-incidence matrices are two matrices, P and Q, with n rows and 2m
    columns.  That is, there is one row for each node, and one column for
    each *directed* edge.  For P, the entry at (n, e) is equal to 1 if node
    n is the source (or tail) of edge e, and 0 otherwise.  For Q, the entry
    at (n, e) is equal to 1 if node n is the target (or head) of edge e,
    and 0 otherwise.

    Params
    ------

    graph (nx.Graph): The graph.

    ordering (str): If 'blocks' (default), the two columns corresponding to
    the i'th edge are placed at i and i+m.  That is, choose an arbitarry
    direction for each edge in the graph.  The first m columns correspond
    to this orientation, while the latter m columns correspond to the
    reversed orientation.  Columns in either block are sorted following
    graph.edges().  If 'consecutive', the first two columns correspond to
    the two orientations of the first edge, the third and fourth row are
    the two orientations of the second edge, and so on.  In general, the
    two columns for the i'th edge are placed at 2i and 2i+1.

    return_ordering (bool): if True, return a function that maps an edge id
    to the column placement.  That is, if ordering=='blocks', return the
    function lambda x: (x, m+x), if ordering=='consecutive', return the
    function lambda x: (2*x, 2*x + 1).  If False, return None.


    Returns
    -------

    P (sparse matrix), Q (sparse matrix), ordering (function or None).


    Notes
    -----

    The nodes in graph must be labeled by consecutive integers starting at
    0.  This function always returns three values, regardless of the value
    of return_ordering.

    """
    numnodes = graph.order()
    numedges = graph.size()

    if ordering == 'blocks':
        src_pairs = lambda i, u, v: [(u, i), (v, numedges + i)]
        tgt_pairs = lambda i, u, v: [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        src_pairs = lambda i, u, v: [(u, 2*i), (v, 2*i + 1)]
        tgt_pairs = lambda i, u, v: [(v, 2*i), (u, 2*i + 1)]

    def make_coo(make_pairs):
        """Make a sparse 0-1 matrix.

        The returned matrix has a positive entry at each coordinate pair
        returned by make_pairs, for all (idx, node1, node2) edge triples.

        """
        coords = list(zip(*(pair
                            for idx, (node1, node2) in enumerate(graph.edges())
                            for pair in make_pairs(idx, node1, node2))))
        data = np.ones(2*graph.size())
        return sparse.coo_matrix((data, coords),
                                 shape=(numnodes, 2*numedges))

    src = make_coo(src_pairs).asformat('csr')
    tgt = make_coo(tgt_pairs).asformat('csr')

    if return_ordering:
        if ordering == 'blocks':
            func = lambda x: (x, numedges + x)
        else:
            func = lambda x: (2*x, 2*x + 1)
        return src, tgt, func
    else:
        return src, tgt, None


def fast_hashimoto(graph, ordering='blocks', return_ordering=False):
    """Make the Hashimoto (aka Non-Backtracking) matrix.

    Params
    ------

    graph (nx.Graph): A NetworkX graph object.

    ordering (str): Ordering used for edges (see `half_incidence`).

    return_ordering (bool): If True, return the edge ordering used (see
    `half_incidence`).  If False, only return the matrix.

    Returns
    -------

    A sparse (csr) matrix.

    """
    sources, targets, ord_func = half_incidence(graph, ordering, return_ordering)
    temp = np.dot(targets.T, sources).asformat('coo')
    temp_coords = set(zip(temp.row, temp.col))

    coords = [(r, c) for r, c in temp_coords if (c, r) not in temp_coords]
    data = np.ones(len(coords))
    shape = 2*graph.size()
    hashimoto = sparse.coo_matrix((data, list(zip(*coords))), shape=(shape, shape))

    if return_ordering:
        return hashimoto.asformat('csr'), ord_func
    else:
        return hashimoto.asformat('csr')


#############################
# 2. Eigenvalue computation #
#############################

def nbeigs(graph, topk, fmt='complex'):
    """Compute the largest non-backtracking eigenvalues of a graph.

    Params
    ------

    graph (nx.Graph): The graph.

    topk (int): The number of eigenvalues to compute.  The maximum number
    of eigenvalues that can be computed is 2*n - 2, where n is the number
    of nodes in graph.  All the other eigenvalues are equal to +-1.

    fmt (str): The format of the return value.  If 'complex', return a list
    of complex numbers, sorted by increasing absolute value.  If '2D',
    return a 2D array of shape (topk, 2), where the first column contains
    the real part of each eigenvalue, and the second column, the imaginary
    part.  If '1D', return an array of shape (2*topk,) made by concatenaing
    the two columns of the '2D' version into one long vector.

    tol (float): Numerical tolerance.  Default 1e-5.

    Returns
    -------

    A list or array with the eigenvalues, depending on the value of fmt.

    """
    # The eigenvalues are left untouched by removing the nodes of degree 1.
    # Moreover, removing them makes the computations faster.  This
    # 'shaving' leaves us with the 2-core of the graph.
    core = shave(graph)
    matrix = pseudo_hashimoto(core)
    if topk > matrix.shape[0] - 1:
        topk = matrix.shape[0] - 2
        print('Computing only {} eigenvalues'.format(topk))

    if topk < 1:
        return np.array([[], []])

    # The eigenvalues are sometimes returned in no particular order, which
    # may yield different feature vectors for the same graph. For example,
    # if a graph has a + ib and a - ib as eigenvalues, the eigenvalue
    # solver may return [..., a + ib, a - ib, ...] in one call and [..., a
    # - ib, a + ib, ...] in another call. To avoid this, we arbitrarily
    # sort the eigenvalues first by absolute value, then by real part, then
    # by imaginary part.
    eigs = sparse.linalg.eigs(matrix, k=topk, return_eigenvectors=False, tol=TOL)
    eigs = sorted(eigs, key=lambda x: x.imag)
    eigs = sorted(eigs, key=lambda x: x.real)
    eigs = np.array(sorted(eigs, key=norm))

    if fmt.lower() == 'complex':
        return eigs

    eigs = np.array([(c.real, c.imag) for c in eigs])
    if fmt.upper() == '2D':
        return eigs

    if fmt.upper() == '1D':
        return eigs.T.flatten()


def shave(graph):
    """Return the 2-core of a graph.

    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.

    """
    core = graph.copy()
    while True:
        to_remove = [node for node, neighbors in core.adj.items()
                     if len(neighbors) < 2]
        core.remove_nodes_from(to_remove)
        if len(to_remove) == 0:
            break
    return core



###########################
# 3. Distance computation #
###########################

def nbdist(graph1, graph2, topk, sigma=1.0, eta=0.0):
    """Compute the non-backtracking spectral distance.

    Let c_j = a_j + i * b_j be the j-th non-backtracking eigenvalue of
    graph1, for j=1,2,..,topk.  We build the vector
        v_1 = (a_1, a_2, ..., a_topk, b_1, b_2, ..., b_topk)
    and compare it to the corresponding vector v_2 coming from graph2 using
    the Euclidean distance.

    Params
    ------

    graph1, graph2 (nx.Graph): The graphs to compare.

    topk (int): The number of eigenvalues to compute and compare.

    sigma (flat): Fine-tuning parameter for number of triangles.  Before
    comparison, we replace v_i with
        v'_i = (sigma*a_1, sigma*a_2, ...,
                b_1/sigma, b_2/sigma, ...)
    for i=1,2.  The larger the sigma, the larger the emphasis on number of
    triangles when comparing two graphs.  Default 1.0 (no emphasis).

    eta (float): Fine-tuning parameter for degree distribution.  Before
    comparison, we replace v_i with
        v'_i = (|c_1|^eta * a_1, |c_2|^eta * a_2, ...,
                |c_1|^eta * b_1, |c_2|^eta * b_2, ...)
    for i=1,2.  The larger the eta, the larger the emphasis on the second
    moment of the degree distribution when comparing two graphs.  Default
    0.0 (no emphasis).

    Returns
    -------

    A real number, the distance between graph1 and graph2.

    Notes
    -----

    If one graph has fewer eigenvalues than requested, the comparison uses
    the most possible eigenvalues.  For more information on the fine tuning
    parameters sigma and eta, see [1].

    References
    ----------

    [1] Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad, **Graph Distance
    from the Topological View of Non-Backtracking Cycles**, preprint,
    arXiv:1807.09592 [cs.SI], (2018).

    """
    def fine_tune(graph):
        """Return fine-tuned eigenvalues."""
        eigs = nbeigs(graph, topk, fmt='complex')
        vals = np.abs(eigs)**eta
        eigs = eigs * vals
        eigs = np.array([(c.real * sigma, c.imag / sigma) for c in eigs])
        return eigs

    eigs1 = fine_tune(graph1)
    eigs2 = fine_tune(graph2)
    min_len = min(eigs1.shape[0], eigs2.shape[0])
    eigs1 = eigs1[:min_len].T.flatten()
    eigs2 = eigs2[:min_len].T.flatten()

    return np.linalg.norm(eigs1 - eigs2)
