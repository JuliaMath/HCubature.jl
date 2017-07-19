# Gauss-Kronrod quadrature rule, via the QuadGK package, since Genz-Malik
# rule does not handle the 1d case.  We will just use a fixed-order (7) G-K rule.

struct GaussKronrod{T<:AbstractFloat}
    x::Vector{T}
    w::Vector{T}
    wg::Vector{T}
end

GaussKronrod(::Type{T}) where {T<:AbstractFloat} =
    GaussKronrod{T}(QuadGK.kronrod(T,7)...)

function (g::GaussKronrod{T})(f, a_::SVector{1}, b_::SVector{1}, norm=vecnorm) where {T}
    a = a_[1]
    b = b_[1]
    c = (a+b)*T(0.5)
    Δ = (b-a)*T(0.5)

    fx⁰ = f(SVector(c))                # f(0)
    I = fx⁰ * g.w[end]
    I′ = fx⁰ * g.wg[end]
    @inbounds for i = 1:length(g.x)-1
        Δx = Δ * g.x[i]
        fx = f(SVector(c + Δx)) + f(SVector(c - Δx))
        I += fx * g.w[i]
        if iseven(i)
            I′ += fx * g.wg[i>>1]
        end
    end
    V = abs(Δ)
    I *= V
    I′ *= V
    return I, norm(I - I′), 1
end
