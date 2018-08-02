# implementation of the n-dimensional cubature rule (n ≥ 2) from
# A. C. Genz and A. A. Malik, "An adaptive algorithm for numeric
# integration over an N-dimensional rectangular region," J. Comput. Appl. Math.,
# vol. 6 (no. 4), 295-302 (1980).

"""
    combos(k, λ, Val{n}())

Return an array of SVector{n} of all n-component vectors
with k components equal to λ and other components equal to zero.
"""
function combos(k::Integer, λ::T, ::Val{n}) where {n, T<:Number}
    combos = Combinatorics.combinations(1:n, k)
    p = Vector{SVector{n,T}}(undef, length(combos))
    v = MVector{n,T}(undef)
    for (i,c) in enumerate(combos)
        v .= 0
        v[c] .= λ
        p[i] = v
    end
    return p
end

"""
    signcombos(k, λ, Val{n}())

Return an array of SVector{n} of all n-component vectors
with k components equal to ±λ and other components equal to zero
(with all possible signs).
"""
function signcombos(k::Integer, λ::T, ::Val{n}) where {n, T<:Number}
    combos = Combinatorics.combinations(1:n, k)
    twoᵏ = 1 << k
    p = Vector{SVector{n,T}}(undef, length(combos) * twoᵏ)
    v = MVector{n,T}(undef)
    for (i,c) in enumerate(combos)
        j = (i-1)*twoᵏ + 1
        v .= 0
        v[c] .= λ
        p[j] = v
        # use a gray code to flip one sign at a time
        graycode = 0
        for s = 1:twoᵏ-1
            graycode′ = s ⊻ (s >> 1)
            graycomp = c[trailing_zeros(graycode ⊻ graycode′) + 1]
            graycode = graycode′
            v[graycomp] = -v[graycomp]
            p[j+s] = v
        end
    end
    return p
end

"""
`GenzMalik{n,T}` holds the points and weights corresponding
to an `n`-dimensional Genz-Malik cubature rule over coordinates
of type `T`.
"""
struct GenzMalik{n,T<:AbstractFloat}
    p::NTuple{4,Vector{SVector{n,T}}} # points for the last 4 G-M weights
    w::NTuple{5,T}  # weights for the 5 terms in the G-M rule
    w′::NTuple{4,T} # weights for the embedded lower-degree rule
end

# cache the Genz-Malik rules so that we don't reconstruct them every time;
# this mainly matters for simple integrands (low-degree polynomials) that
# don't require refinement.
const gmcache = Dict{Tuple{Int,Type}, GenzMalik}()

"""
    GenzMalik(Val{n}(), T=Float64)

Construct an n-dimensional Genz-Malik rule for coordinates of type `T`.
"""
function GenzMalik(v::Val{n}, ::Type{T}=Float64) where {n, T<:AbstractFloat}
    haskey(gmcache, (n,T)) && return gmcache[n,T]::GenzMalik{n,T}

    n < 2 && throw(ArgumentError("invalid dimension $n: GenzMalik rule requires dimension > 2"))

    λ₄ = sqrt(9/T(10))
    λ₂ = sqrt(9/T(70))
    λ₃ = λ₄
    λ₅ = sqrt(9/T(19))

    twoⁿ = 1 << n
    w₁ = twoⁿ * ((12824 - 9120n + 400n^2) / T(19683))
    w₂ = twoⁿ * (980 / T(6561))
    w₃ = twoⁿ * ((1820 - 400n) / T(19683))
    w₄ = twoⁿ * (200 / T(19683))
    w₅ = 6859/T(19683)
    w₄′ = twoⁿ * (25/T(729))
    w₃′ = twoⁿ * ((265 - 100n)/T(1458))
    w₂′ = twoⁿ * (245/T(486))
    w₁′ = twoⁿ * ((729 - 950n + 50n^2)/T(729))

    p₂ = combos(1, λ₂, v)
    p₃ = combos(1, λ₃, v)
    p₄ = signcombos(2, λ₄, v)
    p₅ = signcombos(n, λ₅, v)

    g = GenzMalik{n,T}((p₂,p₃,p₄,p₅), (w₁,w₂,w₃,w₄,w₅), (w₁′,w₂′,w₃′,w₄′))
    gmcache[n,T] = g
    return g
end

countevals(g::GenzMalik{n}) where {n} = 1 + 4n + 2*n*(n-1) + (1<<n)

"""
    genzmalik(f, a, b, norm=vecnorm)

Evaluate `genzmalik::GenzMalik` for the box with min/max corners `a` and `b`
for an integrand `f`.  Returns the estimated integral `I`, the estimated
error `E` (via the given `norm`), and the suggested coordinate `k` ∈ `1:n`
to subdivide next.
"""
function (g::GenzMalik{n,T})(f, a::SVector{n}, b::SVector{n}, norm=vecnorm) where {n,T}
    c = T(0.5).*(a.+b)
    Δ = T(0.5).*abs.(b.-a)
    V = prod(Δ)

    f₁ = f(c)

    f₂ = zero(f₁)
    f₃ = zero(f₁)
    twelvef₁ = 12f₁
    maxdivdiff = zero(norm(f₁))
    divdiff = MVector{n,typeof(maxdivdiff)}(undef)
    for i = 1:n
        p₂ = Δ .* g.p[1][i]
        f₂ᵢ = f(c + p₂) + f(c - p₂)
        p₃ = Δ .* g.p[2][i]
        f₃ᵢ = f(c + p₃) + f(c - p₃)
        f₂ += f₂ᵢ
        f₃ += f₃ᵢ
        # fourth divided difference: f₃ᵢ-2f₁ - 7*(f₂ᵢ-2f₁),
        # where 7 = (λ₃/λ₂)^2 [see van Dooren and de Ridder]
        divdiff[i] = norm(f₃ᵢ + twelvef₁ - 7*f₂ᵢ)
    end

    f₄ = zero(f₁)
    for p in g.p[3]
        f₄ += f(c .+ Δ .* p)
    end

    f₅ = zero(f₁)
    for p in g.p[4]
        f₅ += f(c .+ Δ .* p)
    end

    I = V * (g.w[1]*f₁ + g.w[2]*f₂ + g.w[3]*f₃ + g.w[4]*f₄ + g.w[5]*f₅)
    I′ = V * (g.w′[1]*f₁ + g.w′[2]*f₂ + g.w′[3]*f₃ + g.w′[4]*f₄)
    E = norm(I - I′)

    # choose axis
    kdivide = 1
    δf = E / (10^n * V)
    for i = 1:n
        if (δ = divdiff[i] - maxdivdiff) > δf
            kdivide = i
            maxdivdiff = divdiff[i]
        elseif abs(δ) < δf && Δ[i] > Δ[kdivide]
            kdivide = i
        end
    end

    return I, E, kdivide
end
