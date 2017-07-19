using HCubature, StaticArrays
using Base.Test

@test hcubature(x -> cos(x[1])*cos(x[2]), [0,0], [1,1])[1] ≈ sin(1)^2 ≈
      hcubature(x -> cos(x[1])*cos(x[2]), (0,0), (1,1))[1]
@test hcubature(x -> cos(x[1]), (0,), (1,))[1] ≈ sin(1)
@test hcubature(x -> 1.7, SVector{0,Float64}(), SVector{0,Float64}())[1] == 1.7
