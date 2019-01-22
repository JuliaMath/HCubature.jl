VERSION < v"0.7.0-beta2.199" && __precompile__()
"""
The HCubature module is a pure-Julia implementation of
multidimensional "h-adaptive" integration.  That is, given
an n-dimensional integral

``\\int_{a_1}^{b_1}\\int_{a_2}^{b_2}\\cdots\\int_{a_n}^{b_n} (\\vec{x}) d^n\\vec{x}``

then `hcubature(f, a, b)` computes the integral, adaptively subdividing
the integration volume into smaller and smaller pieces until convergence
is achieved to the desired tolerance (specified by optional `rtol` and
`atol` keyword arguments, described in more detail below.

Because `hcubature` is written purely in Julia, the integrand `f(x)`
can return any vector-like object (technically, any type supporting
`+`, `-`, `*` real, and `norm`: a Banach space).  You can integrate
real, complex, and matrix-valued integrands, for example.
"""
module HCubature

using StaticArrays, LinearAlgebra
import Combinatorics, DataStructures, QuadGK

export hcubature, hquadrature

include("genz-malik.jl")
include("gauss-kronrod.jl")

struct Box{n,T<:Real,TI,TE<:Real}
    a::SVector{n,T}
    b::SVector{n,T}
    I::TI
    E::TE
    kdiv::Int
end
Base.isless(i::Box, j::Box) = isless(i.E, j.E)

cubrule(v::Val{n}, ::Type{T}) where {n,T} = GenzMalik(v, T)
cubrule(::Val{1}, ::Type{T}) where {T} = GaussKronrod(T)

# trivial rule for 0-dimensional integrals
struct Trivial; end
function (::Trivial)(f, a::SVector{0}, b::SVector{0}, norm)
    I = f(a)
    return I, norm(I - I), 1
end
cubrule(::Val{0}, ::Type{T}) where {T} = Trivial()
countevals(::Trivial) = 1

function hcubature_(f, a::SVector{n,T}, b::SVector{n,T}, norm, rtol_, atol, maxevals, initdiv) where {n, T<:AbstractFloat}
    rtol = rtol_ == 0 == atol ? sqrt(eps(T)) : rtol_
    (rtol < 0 || atol < 0) && throw(ArgumentError("invalid negative tolerance"))
    maxevals < 0 && throw(ArgumentError("invalid negative maxevals"))
    initdiv < 1 && throw(ArgumentError("initdiv must be positive"))

    rule = cubrule(Val{n}(), T)
    numevals = evals_per_box = countevals(rule)

    Δ = (b-a) / initdiv
    b1 = initdiv == 1 ? b : a+Δ
    I, E, kdiv = rule(f, a,b1, norm)
    (n == 0 || iszero(prod(Δ))) && return I,E
    firstbox = Box(a,b1, I,E,kdiv)
    boxes = DataStructures.BinaryMaxHeap{typeof(firstbox)}()
    push!(boxes, firstbox)

    ma = MVector(a)
    mb = MVector(b)

    if initdiv > 1 # initial box divided by initdiv along each dimension
        skip = true # skip the first box, which we already added
        @inbounds for c in CartesianIndices(ntuple(i->Base.OneTo(initdiv), Val{n}())) # Val ntuple loops are unrolled
            if skip; skip=false; continue; end
            for i = 1:n
                ma[i] = a[i]+(c[i]-1)*Δ[i]
                mb[i] = c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i]
            end
            x = SVector(ma)
            y = SVector(mb)
            # this is shorter and has unrolled loops, but somehow creates a type instability:
            # x = SVector(ntuple(i -> a[i]+(c[i]-1)*Δ[i], Val{n}()))
            # y = SVector(ntuple(i -> c[i]==initdiv ? b[i] : a[i]+c[i]*Δ[i], Val{n}()))
            box = Box(x,y, rule(f, x,y, norm)...)
            I += box.I; E += box.E; numevals += evals_per_box
            push!(boxes, box)
        end
    end

    (E ≤ max(rtol*norm(I), atol) || numevals ≥ maxevals) && return I,E

    @inbounds while true
        # get box with largest error
        box = pop!(boxes)
        # split box along dimension kdiv
        w = (box.b[box.kdiv] - box.a[box.kdiv]) * T(0.5)
        ma[:] = box.a
        ma[box.kdiv] += w
        a′ = SVector(ma)
        mb[:] = box.b
        mb[box.kdiv] -= w
        b′ = SVector(mb)
        # evaluate and push the two new boxes
        box1 = Box(a′,box.b, rule(f, a′,box.b, norm)...)
        box2 = Box(box.a,b′, rule(f, box.a,b′, norm)...)
        push!(boxes, box1)
        push!(boxes, box2)
        I += box1.I + box2.I - box.I
        E += box1.E + box2.E - box.E
        numevals += 2*evals_per_box
        # convergence test:
        (E ≤ max(rtol*norm(I), atol) || numevals ≥ maxevals) && break
    end

    # roundoff paranoia: re-sum
    I = zero(I)
    E = zero(E)
    for i = 1:length(boxes.valtree)
        I += boxes.valtree[i].I
        E += boxes.valtree[i].E
    end
    return I,E
end

function hcubature_(f, a::SVector{n,T}, b::SVector{n,S},
                    norm, rtol, atol, maxevals, initdiv) where {n, T<:Real, S<:Real}
    F = float(promote_type(T, S))
    return hcubature_(f, SVector{n,F}(a), SVector{n,F}(b), norm, rtol, atol, maxevals, initdiv)
end
hcubature_(f, a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
           norm, rtol, atol, maxevals, initdiv) =
    hcubature_(f, SVector{length(a)}(a), SVector{length(b)}(b), norm, rtol, atol, maxevals, initdiv)
hcubature_(f, a::NTuple{n,<:Real}, b::NTuple{n,<:Real},
           norm, rtol, atol, maxevals, initdiv) where {n} =
    hcubature_(f, SVector{n}(a), SVector{n}(b), norm, rtol, atol, maxevals, initdiv)

"""
    hcubature(f, a, b; norm=norm, rtol=sqrt(eps), atol=0, maxevals=typemax(Int), initdiv=1)

Compute the n-dimensional integral of f(x), where `n == length(a) == length(b)`,
over the hypercube whose corners are given by the vectors (or tuples) `a` and `b`.
That is, dimension `x[i]` is integrated from `a[i]` to `b[i]`.  The
return value of `hcubature` is a tuple `(I, E)` of the estimated integral
`I` and an estimated error `E`.

`f` should be a function `f(x)` that takes an n-dimensional vector `x`
and returns the integrand at `x`.   The integrand can be any type that supports
`+`, `-`, `*` real, and `norm` functions.  For example, the integrand
can be real or complex numbers, vectors, matrices, etcetera.

The integrand `f(x)` will be always be passed an `SVector{n,T}`,
where `SVector` is an efficient vector type defined in the `StaticArrays`
package and `T` is a floating-point type determined by promoting
the endpoint `a` and `b` coordinates to a floating-point type.
(Your integrand `f` should be type-stable: it should always return
a value of the same type, given this type of `x`.)

The integrand will never be evaluated exactly at the boundaries of the
integration volume.  (So, for example, it is possible to have an
integrand that blows up at the boundaries, as long as the integral
is finite, though such singularities will slow convergence.)

The integration volume is adaptively subdivided, using a cubature
rule due to Genz and Malik (1980), until the estimated error `E`
satisfies `E ≤ max(rtol*norm(I), atol)`, i.e. `rtol` and `atol` are
the relative and absolute tolerances requested, respectively.
It also stops if the number of `f` evaluations exceeds `maxevals`.
If neither `atol` nor `rtol` are specified, the
default `rtol` is the square root of the precision `eps(T)`
of the coordinate type `T` described above.
Initially, the volume is divided into `initdiv` segments along each dimension.

The error is estimated by `norm(I - I′)`, where `I′` is an alternative
estimated integral (via an "embedded" lower-order cubature rule.)
By default, the norm function used (for both this and the convergence
test above) is `norm`, but you can pass an alternative norm by
the `norm` keyword argument.  (This is especially useful when `f`
returns a vector of integrands with different scalings.)
"""
hcubature(f, a, b; norm=norm, rtol::Real=0, atol::Real=0,
                   maxevals::Integer=typemax(Int), initdiv::Integer=1) =
    hcubature_(f, a, b, norm, rtol, atol, maxevals, initdiv)

"""
    hquadrature(f, a, b; norm=norm, rtol=sqrt(eps), atol=0, maxevals=typemax(Int), initdiv=1)

Compute the integral of f(x) from `a` to `b`.  The
return value of `hcubature` is a tuple `(I, E)` of the estimated integral
`I` and an estimated error `E`.

The other parameters are the same as [`hcubature`](@ref).  `hquadrature``
is just a convenience wrapper around `hcubature` so that you can work
with scalar `x`, `a`, and `b`, rather than 1-component vectors.

Alternatively, for 1d integrals you can import the [`QuadGK`](@ref) module
and call the [`quadgk`](@ref) function, which provides additional flexibility
e.g. in choosing the order of the quadrature rule.
"""
hquadrature(f, a, b; norm=norm, rtol::Real=0, atol::Real=0,
                     maxevals::Integer=typemax(Int), initdiv::Integer=1) =
    hcubature_(x -> f(x[1]), SVector(a), SVector(b), norm, rtol, atol, maxevals, initdiv)

end # module
