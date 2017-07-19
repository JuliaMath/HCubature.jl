# __precompile__()
module HCubature

using Compat, StaticArrays
import Combinatorics, DataStructures, QuadGK

export hcubature

function combos(k::Integer, λ::T, ::Type{Val{n}}) where {n, T<:Number}
    combos = Combinatorics.combinations(1:n, k)
    p = Array{SVector{n,T}}(length(combos))
    v = MVector{n,T}()
    for (i,c) in enumerate(combos)
        v[:] = 0
        v[c] = λ
        p[i] = SVector(v)
    end
    return p
end

function signcombos(k::Integer, λ::T, ::Type{Val{n}}) where {n, T<:Number}
    combos = Combinatorics.combinations(1:n, k)
    twoᵏ = 1 << k
    p = Array{SVector{n,T}}(length(combos) * twoᵏ)
    v = MVector{n,T}()
    for (i,c) in enumerate(combos)
        j = (i-1)*twoᵏ + 1
        v[:] = 0
        v[c] = λ
        p[j] = copy(v)
        # use a gray code to flip one sign at a time
        graycode = 0
        for s = 1:twoᵏ-1
            graycode′ = s ⊻ (s >> 1)
            graycomp = c[trailing_zeros(graycode ⊻ graycode′) + 1]
            graycode = graycode′
            v[graycomp] = -v[graycomp]
            p[j+s] = SVector(v)
        end
    end
    return p
end

# Genz-Malik rule coordinates
struct GenzMalik{n,T<:AbstractFloat}
    p::NTuple{4,Array{SVector{n,T}}}
    w::NTuple{5,T}
    w′::NTuple{4,T}
end

function GenzMalik(::Type{Val{n}}, ::Type{T}=Float64) where {n, T<:AbstractFloat}
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

    p₂ = combos(1, λ₂, Val{n})
    p₃ = combos(1, λ₃, Val{n})
    p₄ = signcombos(2, λ₄, Val{n})
    p₅ = signcombos(n, λ₅, Val{n})

    return GenzMalik{n,T}((p₂,p₃,p₄,p₅), (w₁,w₂,w₃,w₄,w₅), (w₁′,w₂′,w₃′,w₄′))
end

# evaluate rule g for integration of f from a to b
function (g::GenzMalik{n,T})(f, a::SVector{n}, b::SVector{n}, norm=vecnorm) where {n,T}
    c = (a+b)*T(0.5)
    Δ = (b-a)*T(0.5)

    f₁ = f(c)

    f₂ = zero(f₁)
    f₃ = zero(f₁)
    kdivide = 1
    maxdivdiff = zero(norm(f₁))
    twelvef₁ = 12f₁
    for i = 1:n
        p₂ = Δ .* g.p[1][i]
        f₂ᵢ = f(c + p₂) + f(c - p₂)
        p₃ = Δ .* g.p[2][i]
        f₃ᵢ = f(c + p₃) + f(c - p₃)
        f₂ += f₂ᵢ
        f₃ += f₃ᵢ
        # fourth divided difference: f₂ᵢ-2f₁ - 7*(f₃ᵢ-2f₁),
        # where 7 = (λ₃/λ₂)^2 [see van Dooren and de Ridder]
        divdiff = norm(f₂ᵢ + twelvef₁ - 7 * f₃ᵢ)
        if divdiff > maxdivdiff
            kdivide = i
            maxdivdiff = divdiff
        end
    end

    f₄ = zero(f₁)
    for p in g.p[3]
        f₄ += f(c .+ Δ .* p)
    end

    f₅ = zero(f₁)
    for p in g.p[4]
        f₅ += f(c .+ Δ .* p)
    end

    V = abs(prod(b-a))/(1<<n)
    I = V * (g.w[1]*f₁ + g.w[2]*f₂ + g.w[3]*f₃ + g.w[4]*f₄ + g.w[5]*f₅)
    I′ = V * (g.w′[1]*f₁ + g.w′[2]*f₂ + g.w′[3]*f₃ + g.w′[4]*f₄)
    return I, norm(I - I′), kdivide
end

struct Box{n,T<:Real,TI,TE<:Real}
    a::SVector{n,T}
    b::SVector{n,T}
    I::TI
    E::TE
    kdiv::Int
end
Base.isless(i::Box, j::Box) = isless(i.E, j.E)

function hcubature(f, a::SVector{n,T}, b::SVector{n,T}; norm=vecnorm, rtol=sqrt(eps(T)), atol=zero(T), maxevals::Integer=typemax(Int)) where {n, T<:AbstractFloat}
    rule = GenzMalik(Val{n}, T)
    firstbox = Box(a,b, rule(f, a,b, norm)...)
    boxes = DataStructures.binary_maxheap(typeof(firstbox))
    push!(boxes, firstbox)
    I = firstbox.I
    E = firstbox.E
    numevals = 1
    ma = MVector(a)
    mb = MVector(b)
    while E > max(rtol*norm(I), atol) && numevals < maxevals
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
        numevals += 1
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

function hcubature(f, a::SVector{n,T}, b::SVector{n,S}; kws...) where {n, T<:Real, S<:Real}
    F = float(promote_type(T, S))
    return hcubature(f, SVector{n,F}(a), SVector{n,F}(b); kws...)
end
hcubature(f, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}; kws...) =
    hcubature(f, SVector{length(a)}(a), SVector{length(b)}(b); kws...)
hcubature(f, a::NTuple{n,<:Real}, b::NTuple{n,<:Real}; kws...) where {n} =
    hcubature(f, SVector{n}(a), SVector{n}(b); kws...)

end # module
