kw_params = @with_kw (β=0.95, # discount
    γ=3.0, # CRRA
    α=1.0 / 3.0, # capital share
    δ=0.2, # depreciation
    In=0.443993816237631, # normalizing constant for the mollifier
    smoother=1.0, # scale of the mollifier (how peaked the mollifier is) smoother = 0 you get uniform dist
    nw=160, # cash on hand ditn grid points
    ns=50, # skill distribution grid points
    slo=0.5, # lower bound on skill grid
    shi=1.5, # upper bound on skill
    sigs=0.5, # sigma of log normal in income
    sscale=(shi - slo) / 2.0, # second income shock to mollify actual income
    zlo=0.0, # upper bound on this process
    zhi=2.0, # lower bound on cash on hand
    wlo=0.5, # upper bound on cash on hand
    whi=15.0, # size of w grids
    rhoz=0.95) # persistence of agg tfp

const In = 0.443993816237631
const K = 1.5

kw_settings = @with_kw (tol=1e-5, count=1, maxit=100)