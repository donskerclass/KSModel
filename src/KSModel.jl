module KSModel

using LinearAlgebra
using Plots, Distributions, ForwardDiff, Parameters
using FastTransforms, FastGaussQuadrature, FFTW
using BenchmarkTools

include("params.jl")
include("utils.jl")

export kw_params, kw_settings, In, K, chebpts, grids, mollifier, trunc_lognpdf

end # module