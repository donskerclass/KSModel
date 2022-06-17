using KSModel, Parameters, LinearAlgebra, Plots
using ForwardDiff
global K = 1.5
model_params = kw_params(nw=320)
wgrid, wweights, scheb, sgrid, sweights, MW, MWinv = grids(model_params)
@unpack shi, slo, sigs = model_params
g = trunc_lognpdf.(sgrid, Ref(shi), Ref(slo), Ref(0.0), Ref(sigs)) # density g evaluated on sgrid
L = sum(sweights .* g .* sgrid)

Win = 0.3 * ones(model_params.nw)


(ell, c, μ, K, KF) = findss(model_params,
    K,
    L,
    wgrid,
    sgrid,
    wweights,
    sweights,
    Win, MW, MWinv,
    g)

@unpack α, δ, β, nw, ns, γ, zhi, zlo, smoother, rhoz = model_params

# Make the Jacobian 
JJ = zeros(4 * nw + 2, 8 * nw + 4)

# we will always order things XP YP X Y
# convention is that capital letters generally refer to indices
LMP = 1:nw
LELLP = nw+1:2*nw
KKP = 2 * nw + 1
ZP = 2 * nw + 2
MP = 2*nw+3:3*nw+2
ELLP = 3*nw+3:4*nw+2
LM = 4*nw+3:5*nw+2
LELL = 5*nw+3:6*nw+2
KK = 6 * nw + 3
Z = 6 * nw + 4
M = 6*nw+5:7*nw+4
ELL = 7*nw+5:8*nw+4

# create objects needed for solve.jl
funops = 1:4 # which operators output a function
F1 = 1:nw
F2 = nw+1:2*nw
F3 = 2*nw+1:3*nw
F4 = 3*nw+1:4*nw
F5 = 4*nw+1:4*nw+1
F6 = 4*nw+2:4*nw+2


#Create auxiliary variables
c = min.(ell .^ (-1.0 / γ), wgrid) # Consumption Decision Rule
Wss = MPL(1.0, K, L, model_params)              # Steady state wages
Rss = MPK(1.0, K, L, model_params)              # Steady state Rk
dRdK = -(1 - α) * ((Rss + δ - 1) / K)
dWdK = (α * Wss / K)
dRdZ = Rss + δ - 1.0
dWdZ = Wss
dYdK = α * (K^(α - 1)) * (L^(1 - α))
dYdZ = (K^α) * (L^(1 - α))

function molly(x)
    In = 0.443993816237631
    temp = (-1.0 .+ 2.0 * (x .- zlo) / (zhi .- zlo)) / smoother
    y = ((zhi - zlo) / 2.0) * exp.(min.(-1.0 ./ (1.0 .- temp .^ 2), 2.0))
    y = (y / (In * smoother))
    y = y .* (x .> zlo) .* (x .< zhi)
end

#Euler Equation
cfunc(ell) = min.(ell .^ (-1.0 / γ), wgrid)
mollificand(ell, zp, kp) = (repeat(wgrid', nw, ns) - MPK(zp, kp, L, model_params) * repeat(wgrid - cfunc(ell), 1, nw * ns)) / MPL(zp, kp, L, model_params) - kron(sgrid', ones(nw, nw))
fancyF1(ell, ellp, zp, kp) = ell - β * (MPK(zp, kp, L, model_params) / MPL(zp, kp, L, model_params)) * molly(mollificand(ell, zp, kp)) * kron(sweights .* g, wweights .* (cfunc(ellp) .^ (-γ)))

#KF equation
KFmollificand(ell, zp, kp) = (repeat(wgrid, 1, nw * ns) - MPK(zp, kp, L, model_params) * repeat(wgrid' - cfunc(ell)', nw, ns)) / MPL(zp, kp, L, model_params) - kron(sgrid', ones(nw, nw))
fancyF2(m, lm, lell, z, k) = m - molly(KFmollificand(lell, z, k)) * kron(sweights .* (g / MPL(z, k, L, model_params)), wweights .* lm)

fancyF3(lmp, m) = lmp - m
fancyF4(lellp, ell) = lellp - ell
fancyF5(ell, m, kp) = kp - sum((wgrid - cfunc(ell)) .* (m .* wweights))
fancyF6(z, zp) = zp - rhoz * z


# stack all equilibrium conditions
F(lmp, lellp, kp, zp, mp, ellp, lm, lell, k, z, m, ell) = [fancyF1(ell, ellp, zp, kp)
    fancyF2(m, lm, lell, z, k)
    fancyF3(lmp, m)
    fancyF4(lellp, ell)
    fancyF5(ell, m, kp)
    fancyF6(z, zp)
]
# we will always order things XP YP X Y
# convention is that capital letters generally refer to indices

LMP = 1:nw
LELLP = nw+1:2*nw
KKP = 2 * nw + 1
ZP = 2 * nw + 2
MP = 2*nw+3:3*nw+2
ELLP = 3*nw+3:4*nw+2
LM = 4*nw+3:5*nw+2
LELL = 5*nw+3:6*nw+2
KK = 6 * nw + 3
Z = 6 * nw + 4
M = 6*nw+5:7*nw+4
ELL = 7*nw+5:8*nw+4

Fstack(x) = F(x[LMP], x[LELLP], x[KKP], x[ZP], x[MP], x[ELLP], x[LM], x[LELL], x[KK], x[Z], x[M], x[ELL])

# make jacobian
usex = [μ; ell; K; 1.0; μ; ell; μ; ell; K; 1.0; μ; ell]
JFD = x -> ForwardDiff.jacobian(Fstack, x);
JJ = JFD(usex);

# create q matrix which we will use later
qqq = Matrix(I(nw))
qqq[:, 1] = ones(nw)
(qqq1, rrr1) = qr(qqq)

QW = qqq1[:, 2:end]'; #
Qleft = cat(I(nw), QW, QW, I(nw), [1], [1], dims=(1, 2))
Qx = cat(QW, I(nw), [1], [1], dims=(1, 2))
Qy = cat(QW, I(nw), dims=(1, 2))



@time gx2, hx2, gx, hx = solve(JJ, Qleft, Qx, Qy)

# from now on we only plot things for cash grid points
# with at least mindens = 1e-8 density in steady state
mindens = 1e-8
maxw = findlast(μ .> mindens)
wg = wgrid[1:maxw]
dur = 50 #Periods to compute IRFs

# we will need this to make consumption shock
dcdell = diagm((-1 / γ) * (ell .^ (-(1 / γ) - 1)) .* ((ell .^ (-1 / γ)) .<= wgrid))
mpc = [1; (c[2:end] - c[1:end-1]) ./ (wgrid[2:end] - wgrid[1:end-1])]

# first, shock to aggregate component of income
zshock = 0.01
zIRFxc = zeros(2 * nw + 1, dur) #IRF to income shock, coefficient values
zIRFxc[2*nw+1, 1] = zshock
for t = 1:dur-1
    zIRFxc[:, t+1] = hx * zIRFxc[:, t]
end
zIRFyc = gx * zIRFxc #y response

#Shocks in terms of grid values
zIRFxp = Qx'zIRFxc
zIRFyp = Qy'zIRFyc

zIRFμ = zIRFyp[1:nw, :]
zIRFell = zIRFyp[nw+1:2*nw, :]
zIRFk = zIRFxp[2*nw+1, :]
zIRFz = zIRFxp[2*nw+2, :]
zIRFwags = dWdK * zIRFk + dWdZ * zIRFz
zIRFrags = dRdK * zIRFk + dRdZ * zIRFz
zIRFgdp = dYdK * zIRFk + dYdZ * zIRFz
zIRFcfun = dcdell * zIRFell
zIRFC = zIRFcfun' * (wweights .* μ) + zIRFμ' * (wweights .* c)
zIRFI = zIRFk[2:end] - (1 - δ) * zIRFk[1:end-1]

yss = (K^α) * (L^(1 - α))
css = μ' * (wweights .* c)
iss = δ * K

# make IRF's

if !isempty(ARGS) # give any command line argument to plot
    p1 = plot(1:dur, (zIRFgdp / yss) * 100, xlabel="t", title="gdpdIRF", legend=false)
    savefig(p1, "plots/KS/gdpdIRF.png")

    p2 = plot(1:dur, (zIRFC / css) * 100, xlabel="t", title="aggCIRF", legend=false)
    savefig(p2, "plots/KS/aggCIRF.png")

    p3 = plot(1:dur-1, (zIRFI / iss) * 100, xlabel="t", title="inventoriesIRF", legend=false)
    savefig(p3, "plots/KS/inventoriesIRF.png")

    p4 = plot(1:dur, (zIRFz) * 100, xlabel="t", title="zIRF", legend=false)
    savefig(p4, "plots/KS/zIRF.png")

    p5 = plot(1:dur, (zIRFrags / Rss) * 100, xlabel="t", title="ragsIRF", legend=false)
    savefig(p5, "plots/KS/ragsIRF.png")

    p6 = plot(1:dur, (zIRFwags / Wss) * 100, xlabel="t", title="wagsIRF", legend=false)
    savefig(p6, "plots/KS/wagsIRF.png")

    p9 = plot(wg, c[1:maxw], xlabel="w", title="cSSpolicy", legend=false)
    savefig(p9, "plots/KS/cSSpolicy.png")

    p10 = plot(wg, μ[1:maxw], xlabel="w", title="wealthSSdist", legend=false)
    savefig(p10, "plots/KS/wealthSSdist.png")

    pyplot()

    # 3D plots 
    thingtoplot = zIRFμ[1:maxw, :]
    xgrid = repeat(wg, 1, dur)
    ygrid = repeat((1:dur)', maxw, 1)
    plot(xgrid, ygrid, thingtoplot, st=:surface, xlabel="w", ylabel="t")


    # p7 = surface(xgrid, ygrid, thingtoplot, zaxis=(minimum(thingtoplot), maximum(thingtoplot)), lw=0.25, color=:jet, camera=(20, 60))
    # savefig(p7, "plots/KS/wealthIRFdist.png")

    thingtoplot = zIRFcfun[1:maxw, :]
    xgrid = repeat(wg, 1, dur)
    ygrid = repeat((1:dur)', maxw, 1)
    plot(xgrid, ygrid, thingtoplot, st=:surface, xlabel="w", ylabel="t")

    # p8 = surface(xgrid, ygrid, thingtoplot, zaxis=(minimum(thingtoplot), maximum(thingtoplot)), lw=0.25, color=:jet)
    # savefig(p8, "plots/KS/cfunIRF.png")



    indic = ones(nw, nw)
    indic = indic .* (broadcast(-, 1:nw, (1:nw)') .> 0)

    shock = 1
    μsim = repeat(μ, 1, dur) + shock * zIRFμ
    csim = repeat(c, 1, dur) + shock * zIRFcfun
    Msim = indic * diagm(wweights) * μsim
    Mss = indic * diagm(wweights) * repeat(μ, 1, dur)

    cstep = 0.04
    clo = [1:cstep:2...]
    chi = clo .+ cstep
    #clo[1] = c[1]-eps()
    #chi[end] = c[end]
    cgrid = (chi .+ clo) / 2.0
    nbins = length(clo)
    n10c = zeros(nbins, dur)
    n10s = zeros(nbins, dur)

    for t = 1:dur
        for i = 1:nbins
            imin = findfirst(c .> clo[i])
            imax = findlast(c .<= chi[i])
            n10s[i, :] = Mss[imax, :] - Mss[imin, :]
            n10c[i, :] = Msim[imax, :] - Msim[imin, :]
        end
    end
    qIRFc = (n10c - n10s) / shock

    thingtoplot = qIRFc
    xgrid = repeat(cgrid, 1, dur)
    ygrid = repeat((1:dur)', nbins, 1)
    plot(xgrid, ygrid, thingtoplot, st=:surface, xlabel="w", ylabel="t")


    # p11 = surface(xgrid, ygrid, thingtoplot, xlabel="w", ylabel="t", zaxis=(minimum(thingtoplot), maximum(thingtoplot)), color=:jet)
    # savefig(p11, "plots/KS/cdist.png")
end
