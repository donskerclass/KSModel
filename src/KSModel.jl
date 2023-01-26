module KSModel

using LinearAlgebra
using Plots, Distributions, ForwardDiff, Parameters
using FastTransforms, FastGaussQuadrature, FFTW
using BenchmarkTools

include("params.jl")
include("utils.jl")
include("solve.jl")
include("ss.jl")
include("huggett.jl")

export kw_params, kw_settings, In, chebpts, grids, mollifier, trunc_lognpdf
export dmollifier, MPK, MPL, solve
export sspolicy, findss
export huggett_model, huggett_findss, huggett_sspolicy, huggett_solve

end # module
