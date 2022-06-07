model_params = kw_params()
model_settings = kw_settings()
wgrid, wweights, scheb, sgrid, sweights, MW, MWinv = grids(model_params)

# values from v0.6 original
@test wgrid[50] ≈ 4.968553459119497
@test all(wweights .== 0.090625)
@test scheb[1][12] ≈ 0.6192770208154328
@test scheb[2][5] ≈ 0.008136298979831284
@test sgrid[4] ≈ 0.5092204215044673
@test sweights[50] ≈ 0.00020824656393169535
@test all(diag(MW) .≈ 0.3010398644698074)
@test all(diag(MWinv) .≈ 3.3218191941495987)

@unpack shi, slo, sigs = model_params
g = trunc_lognpdf.(sgrid, Ref(shi), Ref(slo), Ref(0.0), Ref(sigs)) # density g evaluated on sgrid
L = sum(sweights .* g .* sgrid)                   # Aggregate Labor endowment

@test size(g) == (50,)
@test size(sgrid) == (50,)
@test size(sweights) == (50,)
@test size(wgrid) == (160,)
@test size(wweights) == (160,)

@test g[42] ≈ 0.6039220156147933
@test L ≈ 0.9475772259381134

ss = sspolicy(model_params, K, L, wgrid, sgrid, wweights, sweights, Win, g)
(c, ap, Win, KF) = ss
@test c[123] ≈ 2.1224811555977614 atol = 1e-2
@test ap[94] ≈ 6.994317121518172 atol = 1e-2
@test Win[30] ≈ 0.239086336045894 atol = 1e-2
@test KF[123, 123] ≈ 0.9155831206512652 atol = 1e-2
@test KF[10, 1] ≈ 0.8991639067400543 atol = 1e-2
@test KF[1, 10] ≈ 5.2138372905809455e-6 atol = 1e-2