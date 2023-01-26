include("../Huggett.jl")

# values from v0.6 original 
@test wgrid[50] ≈ 6.708053691275167
@test all(wweights .== 0.137)
@test scheb[1][9] ≈ 3.4095389311788624
@test all(diag(MW) .≈ 0.37013511046643494)
@test g[7] ≈ 0.1336446504993422

# steady state objects 
@test_broken ell[76] ≈ 0.049558990867949945
@test_broken c[76] ≈ 4.491990489637991
@test_broken R ≈ 0.9355597245065789
@test m[76] ≈ 2.324614853604984e-11 atol = 1e-10

# solve.jl 
@test_broken gx2[5, 10] ≈ 1.0879965597173848
@test_broken hx2[12, 100] ≈ 0.6669025418833682
@test_broken hx[92, 83] ≈ -0.008244972017033574

@test_broken agIRFc[85, 7] ≈ 0.4831784976013633
