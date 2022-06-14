using KSModel, Parameters, LinearAlgebra
using Test

@testset "KS" begin
    include("KS.jl")
end

@testset "KS_autodiff" begin
    include("KS_autodiff.jl")
end