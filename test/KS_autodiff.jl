include("../KS_autodiff.jl") # run -top level script

# values from v0.6 code
@test ell[50] ≈ 0.25849099100147527 rtol = 1e-3
@test c[10] ≈ 0.9090909090909091 rtol = 1e-3
# @test μ[end] ≈ 1.8526587876946904e-12 rtol = 1e-3 # order -12
@test K ≈ 1.5275456653905812 rtol = 1e-3
@test KF[7, 16] ≈ 0.1660793825107261 rtol = 1e-3

@test gx2[40, 24] ≈ 0.014640341677680153 rtol = 1e-3
# @test hx2[end, 4] ≈ -7.709133088505682e-16 rtol = 1e-3 # order -16
@test gx[13, 169] ≈ -0.00020403263741761678 rtol = 1e-3
# @test hx[21, 441] ≈ -3.2038320022138855e-10 rtol = 1e-3 # order -10
