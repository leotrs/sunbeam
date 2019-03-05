"""
sunbeam.py
-----------

(S)pectral (N)on-(B)acktracking (E)mbedding (A)nd Pseudo-(M)etric. This
module contains functions related to using the eigenvalues of the
non-backtracking matrix to perform graph mining tasks such as graph
distance and graph embedding. The functions found here are grouped under
the following headings, based on what they do.

## Matrix Representations

The function `fast_hashimoto` computes the Hashimoto or non-backtracking
matrix of a graph, while `pseudo_hashimoto` computes a block matrix whose
eigenvalues are also eigenvalues of the Hashimoto matrix.  The function
`half_incidence` returns the half-incidence matrices of the graph.  These
are useful when computing the Hashimoto matrix or its eigendecomposition
efficiently.  Building the pseudo-Hashimoto matrix is always faster than
building the Hashimoto matrix.  Building the former takes as long as
building the adjacency matrix of the graph.  Building the latter scales as
the second moment of the degree distribution of the graph (i.e., graphs
with heavy-tail degree distributions will take longer).

## Eigendecomposition

The function `nbvals` returns the non-backtracking eigenvalues of a graph.
The eigenvalues may be returned as complex numbers, as a 2D array, or as a
1D 'unrolled' feature vector.  Different formats are convenient for
different purposes.  In computing the eigenvalues, we first obtain the
2-core of the graph by a process known as 'shaving'.  The function `nbvecs`
returns the eigenvectors of the non-backtracking matrix, which can be used
for visualization and as an embedding technique.  The first eigenvector
always has real entries, but there is no guarantee for the n-th eigenvector
for n>1.  You can choose whether to apply a projection to get a real-valued
embedding. `nbvecs` can be used to return both eigenvectors and
eigenvalues.

## Distance

The function `nbd` takes two graphs and returns the distance between
their non-backtracking eigenvalue spectra.  It also offers the optional
fine-tuning parameters sigma and eta.  These are used to make `nbdist` more
versatile and able to emphasize triangles or degree distribution.

## Embedding

The function `nbed` returns the 2-dimensional, real, embedding of a graph.
Visualization is available through `visualize_nbed`.

## Notes

This library assumes that the nodes of every graph are labeled by
consecutive integers starting with 0.

## References

[1] Leo Torres, P. Suárez Serrato, and T. Eliassi-Rad, **Graph Distance
    from the Topological View of Non-Backtracking Cycles**, preprint,
    arXiv:1807.09592 [cs.SI], (2018).

Autor: leotrs (leo@leotrs.com)

"""

import numpy as np
from ot import emd2
import networkx as nx
import scipy.sparse as sparse
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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
    is the adjacency matrix.  The eigenvalues of B' are always eigenvalues
    of B, the non-backtracking or Hashimoto matrix.

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
    reversed orientation.  Columns are sorted following graph.edges().  If
    'consecutive', the first two columns correspond to the two orientations
    of the first edge, the third and fourth row are the two orientations of
    the second edge, and so on.  In general, the two columns for the i'th
    edge are placed at 2i and 2i+1.

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
        return src, tgt


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
    if return_ordering:
        sources, targets, ord_func = half_incidence(graph, ordering, return_ordering)
    else:
        sources, targets = half_incidence(graph, ordering, return_ordering)
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
# 2. Eigendecomposition     #
#############################

def nbvecs(graph, topk, return_vals=False, tol=TOL):
    """Compute the unit eigenvectors corresponding to the largest eigenvalues.

    By convention, return the eigenvectors all of whose first entries have
    positive real parts.  This is done to avoid the arbitrariness in the
    signs of the eigenvectors, since whenever v is an eigenvector, -v also
    is one.

    Params
    ------

    graph (nx.Graph): The graph.

    topk (int): The nunber of eigenvectors to compute.

    return_vals (bool): Whether to return eigenvalues too.  Default False.

    tol (float): Numerical tolerance.  Default 1e-5.

    Returns
    -------

    Array of size (topk, 2*graph.size()).

    """
    matrix = fast_hashimoto(graph)
    vals, vecs = sparse.linalg.eigs(matrix, k=topk, return_eigenvectors=True, tol=tol)

    for i in range(topk):
        if vecs[0, i] < 0:
            vecs[:, i] *= -1

    if return_vals:
        return vecs, vals
    else:
        return vecs


def nbvals(graph, topk='automatic', batch=100, fmt='complex', tol=TOL):
    """Compute the largest-magnitude non-backtracking eigenvalues.

    Params
    ------

    graph (nx.Graph): The graph.

    topk (int or 'automatic'): The number of eigenvalues to compute.  The
    maximum number of eigenvalues that can be computed is 2*n - 4, where n
    is the number of nodes in graph.  All the other eigenvalues are equal
    to +-1. If 'automatic', return all eigenvalues whose magnitude is
    larger than the square root of the largest eigenvalue.

    batch (int): If topk is 'automatic', compute this many eigenvalues at a
    time until the condition is met.  Must be at most 2*n - 4; default 100.

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
    if not isinstance(topk, str) and topk < 1:
        return np.array([[], []])

    # The eigenvalues are left untouched by removing the nodes of degree 1.
    # Moreover, removing them makes the computations faster.  This
    # 'shaving' leaves us with the 2-core of the graph.
    core = shave(graph)
    matrix = pseudo_hashimoto(core)
    if not isinstance(topk, str) and topk > matrix.shape[0] - 1:
        topk = matrix.shape[0] - 2
        print('Computing only {} eigenvalues'.format(topk))

    if topk == 'automatic':
        batch = min(batch, 2*graph.order() - 4)
        if 2*graph.order() - 4 < batch:
            print('Using batch size {}'.format(batch))
        topk = batch
    eigs = lambda k: sparse.linalg.eigs(matrix, k=k, return_eigenvectors=False, tol=tol)
    count = 1
    while True:
        vals = eigs(topk*count)
        largest = np.sqrt(abs(max(vals, key=abs)))
        if abs(vals[0]) <= largest or topk != 'automatic':
            break
        count += 1
    if topk == 'automatic':
        vals = vals[abs(vals) > largest]

    # The eigenvalues are returned in no particular order, which may yield
    # different feature vectors for the same graph.  For example, if a
    # graph has a + ib and a - ib as eigenvalues, the eigenvalue solver may
    # return [..., a + ib, a - ib, ...] in one call and [..., a - ib, a +
    # ib, ...] in another call.  To avoid this, we sort the eigenvalues
    # first by absolute value, then by real part, then by imaginary part.
    vals = sorted(vals, key=lambda x: x.imag)
    vals = sorted(vals, key=lambda x: x.real)
    vals = np.array(sorted(vals, key=np.linalg.norm))

    if fmt.lower() == 'complex':
        return vals

    vals = np.array([(z.real, z.imag) for z in vals])
    if fmt.upper() == '2D':
        return vals

    if fmt.upper() == '1D':
        return vals.T.flatten()


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

def nbd(graph1, graph2, topk='automatic', metric='EMD', vals=None, sigma=1.0,
        eta=0.0):
    """Compute the non-backtracking spectral distance (NBD).

    NBD uses the non-backtracking eigenvalues for graph comparison. The
    metric used to compare the two sets of eigenvalues is controlled by
    parameter `metric`.  Assume lambda_i, i=1,...,n are the eigenvalues of
    graph1, and mu_j, j=1,...,m are the eigenvalues of graph2.

    Params
    ------

    graph1, graph2 (nx.Graph): The graphs to compare.

    topk (int or 'automatic'): The number of eigenvalues to compute and
    compare.  If 'automatic' (default), use the eigenvalues whose magnitude
    is larger than sqrt(lambda_1) where lambda_1 is the largest eigenvalue
    of graph1.  For graph2, use sqrt(mu_1) instead.  Note this may yield
    different number of eigenvalues for each graph.  See parameter `metric`
    to see how this is handled by each distance.  If int, compute the same
    number of eigenvalues of both graphs.

    metric (str): Determines the distance used to compare eigenvalues. If
    'emd' (default) use earth-mover distance (aka Wasserstein).  If
    Hausdorff, use $max(max_j min_i d(lambda_i, mu_j), max_i min_j
    d(lambda_i, mu_j))$.  If 'euclidean', use Euclidean distance between
    the vectors $(lambda_1,..., lambda_n)$ for graph1 and $(mu_1,...,
    mu_m)$ for graph2.  For Euclidean, if n != m, use the smallest number
    of eigenvalues for both graphs.

    vals (tuple): a 2-tuple containing precomputed eigenvalues in any shape
    output by nbvals.  If not None, use these eigenvalues instead of
    computing them again.  In that case, and if `topk` is 'automatic', only
    use the outer eigenvalues among these precomputed ones.

    sigma (flat): Fine-tuning parameter for number of triangles.  Before
    comparison, we replace v_i with $v'_i = (sigma*a_1, sigma*a_2, ...,
    b_1/sigma, b_2/sigma, ...)$ for i=1,2.  The larger the sigma, the
    larger the emphasis on number of triangles when comparing two graphs.
    Default 1.0 (no emphasis).

    eta (float): Fine-tuning parameter for degree distribution.  Before
    comparison, we replace v_i with $v'_i = (|c_1|^eta * a_1, |c_2|^eta *
    a_2, ..., |c_1|^eta * b_1, |c_2|^eta * b_2, ...)$ for i=1,2.  The
    larger the eta, the larger the emphasis on the second moment of the
    degree distribution when comparing two graphs.  Default 0.0 (no
    emphasis).

    Returns
    -------

    A real number, the distance between graph1 and graph2.

    """
    if vals is None:
        vals1 = nbvals(graph1, topk, fmt='2D')
        vals2 = nbvals(graph2, topk, fmt='2D')
    else:
        vals1, vals2 = vals

    def fine_tune(vals):
        """Return fine-tuned eigenvalues."""
        abs_vals = np.array([np.abs(v) for v in vals])**eta
        eigs = vals * abs_vals
        eigs = np.array([(r * sigma, i / sigma) for r, i in eigs])
        return eigs

    if not np.allclose(eta, 0) or not np.allclose(sigma, 1):
        vals1 = fine_tune(vals1)
        vals2 = fine_tune(vals2)

    if metric.lower() == 'emd':
        mass = lambda num: np.ones(num) / num
        vals_dist = distance_matrix(vals1, vals2)
        result = emd2(mass(vals1.shape[0]), mass(vals2.shape[0]), vals_dist)

    elif metric.lower() == 'hausdorff':
        vals_dist = distance_matrix(vals1, vals2)
        term1 = vals_dist.min(axis=0).max()
        term2 = vals_dist.min(axis=1).max()
        result = max([term1, term2])

    elif metric.lower() == 'euclidean':
        min_len = min(vals1.shape[0], vals2.shape[0])
        vals1 = vals1[:min_len].T.flatten()
        vals2 = vals2[:min_len].T.flatten()
        print(vals1 - vals2)
        result = np.linalg.norm(vals1 - vals2)

    return result


###########################
# 4. Embedding            #
###########################

def nbed(graph, projection='automatic', normalize=True, tol=TOL):
    """Compute the non-backtracking embedding dimensions.

    Params
    ------

    graph (nx.Graph): The graph.

    projection (function or 'automatic'): The function to convert a complex
    number into a real number.  If function, must accept two complex
    numbers as parameters.  If 'automatic', use the function f(a, b) =
    Re(a*b) = a.real * b.real - a.imag * b.imag, where a is an entry of the
    eigenvector, b is the corresponding eigenvalue, and Re(.) takes the
    real part of a complex number.

    normalize (bool): Whether to normalize the eigenvectors to be unit
    length after the projection has been applied.  Default True.

    tol (float): Numerical tolerance.  Default 1e-5.

    Returns
    -------

    Array of size (topk, 2*graph.size()).

    """
    emb, vals = nbvecs(graph, 2, return_vals=True, tol=tol)

    if projection == 'automatic':
        emb = emb.dot(np.diag(vals)).real

    else:
        emb = np.array([[projection(row[i], vals[i]) for i in range(len(row))]
                        for row in emb])

    if normalize:
        normalization = np.linalg.norm(emb, axis=0)
        emb = emb / normalization

    return emb


def edge_degrees(graph, endpoint='source'):
    """Return the edge of a degree.

    Params
    ------

    graph (nx.Graph): The graph.

    endpoint ('source' or 'target'): For the edge (u -> v), whether to
    return the degree of u ('source') or v ('target').

    """
    if endpoint.lower() not in {'source', 'target'}:
        raise ValueError('endpoint must be one of "source" or "target"')

    edges = graph.edges()
    deg_dict = graph.degree()
    degrees_src = np.array([deg_dict[src] for src, _ in edges])
    degrees_tgt = np.array([deg_dict[tgt] for _, tgt in edges])

    if endpoint.lower() == 'source':
        return np.hstack([degrees_src, degrees_tgt])
    if endpoint.lower() == 'target':
        return np.hstack([degrees_tgt, degrees_src])


def visualize_nbed(graph, emb=None, color='source', log=False):
    """Scatter plot of NBED of a graph.

    In the plot, each point corresponds to a directed edge of the graph.

    Params
    ------

    graph (nx.Graph): The graph.

    emb (array): The NBED of graph, as output by `nbed`.  If None
    (default), compute it by calling `nbed`.

    color ('source' or 'target'): Whether to color the point corresponding
    to the edge (u -> v) by the degree of u ('source') or v ('target').

    log (bool): Whether to take the logarithm of the degree before coloring
    each point. Value of True (default) is recommended for networks with
    heavy-tail degree distribution.

    """
    if emb is None:
        emb = nbed(graph)

    colors = edge_degrees(graph, endpoint=color)
    if log:
        colors = np.log(colors)
    colors = colors / colors.max()

    plt.figure()
    plt.scatter(emb.T[0], emb.T[1], c=colors, cmap=get_cmap('viridis'))
    plt.xlabel('Second eigenvector')
    plt.ylabel('First eigenvector')
    plt.show()
