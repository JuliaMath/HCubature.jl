# HCubature

[![Build Status](https://travis-ci.org/stevengj/HCubature.jl.svg?branch=master)](https://travis-ci.org/stevengj/HCubature.jl)

[![Coverage Status](https://coveralls.io/repos/stevengj/HCubature.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/stevengj/HCubature.jl?branch=master)

[![codecov.io](http://codecov.io/github/stevengj/HCubature.jl/coverage.svg?branch=master)](http://codecov.io/github/stevengj/HCubature.jl?branch=master)

The HCubature module is a pure-Julia implementation of
multidimensional "h-adaptive" integration.  That is, given
an n-dimensional integral

![n-dimensional integral](integral.png)


then `hcubature(f, a, b)` computes the integral, adaptively subdividing
the integration volume into smaller and smaller pieces until convergence
is achieved to the desired tolerance (specified by optional `rtol` and
`atol` keyword arguments, described in more detail below.

## Algorithm

The algorithm of `hquadrature` is based on the one describe in:

* A. C. Genz and A. A. Malik, "[An adaptive algorithm for numeric integration over an N-dimensional rectangular region](https://doi.org/10.1016/0771-050X(80)90039-X)," *J. Comput. Appl. Math.*, vol. 6 (no. 4), 295-302 (1980).

## Author and Copyright

HCubature was written by [Steven G. Johnson
(SGJ)](http://math.mit.edu/~stevenj/), and is free/open-source
software under the [MIT/expat license](LICENSE.md).

SGJ also wrote an earlier [C implementation of a similar
algorithm](http://ab-initio.mit.edu/wiki/index.php/Cubature) that is
also callable from Julia via the [Cubature.jl
package](https://github.com/stevengj/Cubature.jl).  The HCubature
package is a from-scratch re-implementation, not a translation, of
this code, both to take advantage of unique features of Julia and to
eliminate licensing restrictions arising from the use of C code taken
from the [HIntLib library](http://mint.sbg.ac.at/HIntLib/).   (In both
cases, the original DCUHRE Fortran code of Genz was not examined, only
the mathematical description in the papers.)
