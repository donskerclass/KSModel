using KSModel, Parameters, LinearAlgebra
using Test

@testset "KS" begin
    include("KS.jl")
    include("Huggett.jl")
end

# these take a long time to run
# @testset "KS_autodiff" begin
#     include("KS_autodiff.jl")
# end