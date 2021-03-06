Higher order tensor decomposition
=================================

Implements the symmetric higher order power method (S-HOPM) and uses it to find **non-negative** rank-one decompositions of **symmetric** higher order (real) tensors.
This is a very specific case of one possible generalization of the eigendecomposition of a matrix.

Note that apart from being limited to non-negative decompositions of symmetric tensors, the current version is also limited to tensors based on a 2D vector space (there is no limit on the order/degree of the tensors).

The use of the S-HOPM and its convergence properties is based on:

> On the Best Rank-1 Approximation of Higher-Order Supersymmetric Tensors *SIAM Journal on Matrix Analysis and Applications*, Vol. 23, No. 3. (January 2002), pp. 863-884, [doi:10.1137/s0895479801387413](http://dx.doi.org/10.1137/s0895479801387413) by Eleftherios Kofidis and Phillip A. Regalia
