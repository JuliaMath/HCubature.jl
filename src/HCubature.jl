# __precompile__()
module HCubature

using Compat, StaticArrays
import Combinatorics, DataStructures, QuadGK

export hcubature

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

cubrule(::Type{Val{n}}, ::Type{T}) where {n,T} = GenzMalik(Val{n}, T)
cubrule(::Type{Val{1}}, ::Type{T}) where {T} = GaussKronrod(T)

# trivial rule for 0-dimensional integrals
struct Trivial; end
function (::Trivial)(f, a::SVector{0}, b::SVector{0}, norm)
    I = f(a)
    return I, norm(I - I), 1
end
cubrule(::Type{Val{0}}, ::Type{T}) where {T} = Trivial()

function hcubature_(f, a::SVector{n,T}, b::SVector{n,T}, norm, rtol, atol, maxevals) where {n, T<:AbstractFloat}
    rule = cubrule(Val{n}, T)
    firstbox = Box(a,b, rule(f, a,b, norm)...)
    boxes = DataStructures.binary_maxheap(typeof(firstbox))
    push!(boxes, firstbox)
    I = firstbox.I
    E = firstbox.E
    n == 0 && return I, E # @inbounds below is wrong for n=0
    numevals = 1
    ma = MVector(a)
    mb = MVector(b)
    @inbounds while E > max(rtol*norm(I), atol) && numevals < maxevals
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

hcubature(f, a::SVector{n,T}, b::SVector{n,T};
          norm=vecnorm, rtol::Real=sqrt(eps(T)), atol::Real=zero(T), maxevals::Integer=typemax(Int)) where {n, T<:AbstractFloat} =
    hcubature_(f, a, b, norm, rtol, atol, maxevals)
function hcubature(f, a::SVector{n,T}, b::SVector{n,S}; kws...) where {n, T<:Real, S<:Real}
    F = float(promote_type(T, S))
    return hcubature(f, SVector{n,F}(a), SVector{n,F}(b); kws...)
end
hcubature(f, a::AbstractVector{<:Real}, b::AbstractVector{<:Real}; kws...) =
    hcubature(f, SVector{length(a)}(a), SVector{length(b)}(b); kws...)
hcubature(f, a::NTuple{n,<:Real}, b::NTuple{n,<:Real}; kws...) where {n} =
    hcubature(f, SVector{n}(a), SVector{n}(b); kws...)

end # module
