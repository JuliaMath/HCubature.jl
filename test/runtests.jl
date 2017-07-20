using HCubature, StaticArrays
using Base.Test

@test hcubature(x -> cos(x[1])*cos(x[2]), [0,0], [1,1])[1] ≈ sin(1)^2 ≈
      hcubature(x -> cos(x[1])*cos(x[2]), (0,0), (1,1))[1]
@test hcubature(x -> cos(x[1]), (0,), (1,))[1] ≈ sin(1)
@test hcubature(x -> 1.7, SVector{0,Float64}(), SVector{0,Float64}())[1] == 1.7

# function wrapper for counting evaluations
const gcnt = Ref(0)
cnt(f) = x -> begin
      gcnt[] += 1
      f(x)
end

let f(x) = sin(x[1] + 3*sin(2*x[2] + 4*sin(3*x[3])))
      gcnt[] = 0
      @test hcubature(cnt(f), (0,0,0),(3,3,3), rtol=1e-6)[1] ≈ -4.78802790509727 rtol=1e-6
      @test 2_400_000 < gcnt[] < 2_500_000
end

@test hcubature(x -> cis(x[1]), (0,), (1,))[1] ≈ (cis(1)-1)/im
@test hcubature(x -> cis(x[1]+x[2]), (0,-1), (1,2))[1] ≈ -(cis(1)-1)*(cis(2)-cis(-1))
