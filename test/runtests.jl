using HCubature, StaticArrays
using Base.Test

@test hcubature(x -> cos(x[1])*cos(x[2]), SVector(0.,0.), SVector(1.,1.))[1] ≈ sin(1)^2
