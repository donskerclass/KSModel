include("../KS.jl") # run -top level script

# values from v0.6 original
@test wgrid[50] ≈ 4.968553459119497
@test all(wweights .== 0.090625)
@test scheb[1][12] ≈ 0.6192770208154328
@test scheb[2][5] ≈ 0.008136298979831284
@test sgrid[4] ≈ 0.5092204215044673
@test sweights[50] ≈ 0.00020824656393169535
@test all(diag(MW) .≈ 0.3010398644698074)
@test all(diag(MWinv) .≈ 3.3218191941495987)

@test size(g) == (50,)
@test size(sgrid) == (50,)
@test size(sweights) == (50,)
@test size(wgrid) == (160,)
@test size(wweights) == (160,)

@test g[42] ≈ 0.6039220156147933
@test L ≈ 0.9475772259381134

@test ell[4] ≈ 0.5543566792109383 atol = 1e-4
# redefined in KS.jl, this is before
# @test c[123] ≈ 2.1228985161434397 atol = 1e-5
@test μ[end] ≈ 1.553272970569121e-12 atol = 1e-14
@test K ≈ 1.5016671850875944 atol = 1e-4
@test KF[42, 42] ≈ 0.9729289780918629 atol = 1e-5

@test Wss ≈ 0.7772518534395161 atol = 1e-4
@test Rss ≈ 1.0452294897469283 atol = 1e-5

@test KFmollificand[120, 1200] ≈ 6.274941207987553 atol = 1e-4
@test μRHS[80] ≈ 0.0005812823286142891 atol = 1e-6

@test ξ[10, 10] ≈ 0.4327055974033748 atol = 1e-4
@test Γ[10, 10] ≈ 0.4168567964380295 atol = 1e-4
@test AA[10, 10] ≈ 0.8373277199773778 atol = 1e-4
@test ζ[10, 10] ≈ 0.8691627302017118 atol = 1e-4

@test dRdK ≈ -0.10887155501238716 atol = 1e-4
@test dWdK ≈ 0.17253193517066445 atol = 1e-4
@test dRdZ ≈ 0.24523099877212884 atol = 1e-4
@test dWdZ ≈ 0.7772494620321403 atol = 1e-4
@test dYdK ≈ 0.24523099877212895 atol = 1e-4
@test dYdZ ≈ 1.1047558336414598 atol = 1e-4

@test JJ[F1, KKP][end] ≈ 0.048470986802242724 atol = 1e-4
@test JJ[F1, ZP][80] ≈ -0.002678746846465019 atol = 1e-4
@test JJ[F1, ELLP][160, 160] ≈ 0.060662867280596146 atol = 1e-4
@test JJ[F1, ELL][21, 21] ≈ -1.1725041713601696 atol = 1e-4
@test JJ[F2, LM][21, 32] ≈ 0.0008577434007329157 atol = 1e-4
@test JJ[F2, LELL][81, 91] ≈ -3.53932806144103e-5 atol = 1e-4
@test JJ[F2, KK][end] ≈ -7.22791431519007e-12 atol = 1e-4
@test JJ[F5, M][end] ≈ -1.1370053471928228 atol = 1e-4

@test Qy[1, 3] ≈ 0.006269623282193984 atol = 1e-4
@test Qx[1, 1] ≈ 0.006269623282193937 atol = 1e-4

@test gx2[5, 6] ≈ 0.024480511351594854 atol = 1e-5
@test hx2[123, 321] ≈ -2.0419701017503358e-7 atol = 1e-6
@test gx[50, 50] ≈ 0.08698854910415967 atol = 1e-5
@test gx[17, 71] ≈ -0.0005336164864925171 atol = 1e-7