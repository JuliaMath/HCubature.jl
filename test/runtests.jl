using HCubature, StaticArrays
using Test

@testset "simple" begin
      @test hcubature(x -> cos(x[1])*cos(x[2]), [0,0], [1,1])[1] ≈ sin(1)^2 ≈
            @inferred(hcubature(x -> cos(x[1])*cos(x[2]), (0,0), (1,1)))[1]
      @test @inferred(hcubature(x -> cos(x[1]), (0,), (1,)))[1] ≈ sin(1) ≈
            @inferred(hquadrature(cos, 0, 1))[1]
      @test @inferred(hcubature(x -> cos(x[1]), (0.0f0,), (1.0f0,)))[1] ≈ sin(1.0f0)
      @test @inferred(hcubature(x -> 1.7, SVector{0,Float64}(), SVector{0,Float64}()))[1] == 1.7
end

# function wrapper for counting evaluations
const gcnt = Ref(0)
cnt(f) = x -> begin
      gcnt[] += 1
      f(x)
end

@testset "big3d" begin
      let f(x) = sin(x[1] + 3*sin(2*x[2] + 4*sin(3*x[3])))
            gcnt[] = 0
            @test hcubature(cnt(f), (0,0,0),(3,3,3), rtol=1e-6)[1] ≈ -4.78802790509727 rtol=1e-6
            @test 2_400_000 < gcnt[] < 2_500_000
            gcnt[] = 0
            hcubature(cnt(f), (0,0,0),(3,3,3), maxevals=1000)
            @test gcnt[] < 1070
      end
end

@testset "complex" begin
      @test hcubature(x -> cis(x[1]), (0,), (1,))[1] ≈ (cis(1)-1)/im
      @test hcubature(x -> cis(x[1]+x[2]), (0,-1), (1,2))[1] ≈ -(cis(1)-1)*(cis(2)-cis(-1))
end

@testset "countevals" begin
      let g = HCubature.GaussKronrod(Float64)
            @test HCubature.countevals(g) == 1 + 2length(g.w)
      end
      for n = 2:10
            let g = HCubature.GenzMalik(Val{n}(), Float64)
                  @test HCubature.countevals(g) == 1 + 4length(g.p[1]) + length(g.p[3]) + length(g.p[4])
            end
      end
      @test HCubature.countevals(HCubature.Trivial()) == 1
end

@testset "axischoosing" begin
    #Issue #4
    let f = x -> 1.0 + (x[1] * x[3] * sin(x[2]))^2
        @test hcubature(f, (0.0,0.0,-0.2), (0.2,2π,0.2), rtol=1e-6)[1] ≈ (22502//140625)*π rtol=1e-6
    end
end

@testset "initdiv" begin
      for initdiv = 1:5
            @test sin(1)^2 ≈ hcubature(x -> cos(x[1])*cos(x[2]), [0,0], [1,1], initdiv=initdiv)[1]
            @test sin(1) ≈ @inferred(hquadrature(cos, 0, 1, initdiv=initdiv))[1]
      end
end