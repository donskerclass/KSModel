using KSModel, Parameters, LinearAlgebra, Plots
global K = 1.5
model_params = kw_params()
wgrid, wweights, scheb, sgrid, sweights, MW, MWinv = grids(model_params)
@unpack shi, slo, sigs = model_params
g = trunc_lognpdf.(sgrid, Ref(shi), Ref(slo), Ref(0.0), Ref(sigs)) # density g evaluated on sgrid
L = sum(sweights .* g .* sgrid)                   # Aggregate Labor endowment

# guess for the W function W(w) = beta R E u'(c_{t+1})
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

# Make the Jacobian 
@unpack α, δ, β, nw, ns, γ, zhi, zlo, smoother, rhoz = model_params
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


# Create auxiliary variables
c = min.(ell .^ (-1.0 / γ), wgrid) # Consumption Decision Rule
Wss = MPL(1.0, K, L, model_params)              # Steady state wages
Rss = MPK(1.0, K, L, model_params)              # Steady state Rk

#KF equation
KFmollificand = (repeat(wgrid, 1, nw * ns) - Rss * repeat(wgrid' - c', nw, ns)) / Wss - kron(sgrid', ones(nw, nw))
μRHS = mollifier.(KFmollificand, zhi, zlo, smoother) * kron(sweights .* (g / Wss), wweights .* μ)

ξ = zeros(nw, nw)
Γ = zeros(nw, nw)
AA = zeros(nw, nw)
ζ = zeros(nw, nw)
for iw = 1:nw
    for iwp = 1:nw
        sumnsξ = 0.0
        sumnsΓ = 0.0
        sumnsAA = 0.0
        sumnsζ = 0.0
        for iss = 1:ns
            mf = (wgrid[iwp] - Rss * (wgrid[iw] - c[iw])) / Wss - sgrid[iss]
            dm = dmollifier(mf, zhi, zlo, smoother)
            mm = mollifier(mf, zhi, zlo, smoother)
            sumnsξ += ((c[iwp] .^ (-γ)) / Wss) * dm * g[iss] * sweights[iss]
            sumnsΓ += ((c[iwp] .^ (-γ)) / Wss) * mm * g[iss] * sweights[iss]
            sumnsAA += (1.0 / Wss) * mm * g[iss] * sweights[iss]
            sumnsζ += (1.0 / Wss) * dm * g[iss] * sweights[iss]
        end
        ξ[iw, iwp] = sumnsξ
        Γ[iw, iwp] = sumnsΓ
        AA[iw, iwp] = sumnsAA
        ζ[iw, iwp] = sumnsζ
    end
end


dRdK = -(1 - α) * ((Rss + δ - 1) / K)
dWdK = (α * Wss / K)
dRdZ = Rss + δ - 1.0
dWdZ = Wss
dYdK = α * (K^(α - 1)) * (L^(1 - α))
dYdZ = (K^α) * (L^(1 - α))
ellRHS = β * Rss * Γ * wweights

# Fill in Jacobian

#Euler equation

JJ[F1, KKP] = (β * Γ * wweights - β * (Rss / Wss) * ((wgrid - c) .* (ξ * wweights))) * dRdK - (1.0 / Wss) * (ellRHS + ((β * Rss) / Wss) * ξ .* (repeat(wgrid', nw, 1) - Rss * repeat(wgrid - c, 1, nw)) * wweights) * dWdK

JJ[F1, ZP] = (β * Γ * wweights - β * (Rss / Wss) * ((wgrid - c) .* (ξ * wweights))) * dRdZ - (1.0 / Wss) * (ellRHS + ((β * Rss) / Wss) * ξ .* (repeat(wgrid', nw, 1) - Rss * repeat(wgrid - c, 1, nw)) * wweights) * dWdZ

JJ[F1, ELLP] = β * Rss * Γ * diagm((wweights .* (ell .^ (-(1.0 + γ) / γ)) .* (ell .^ (-1.0 / γ) .<= wgrid)) ./ c)

JJ[F1, ELL] = -(I(nw) + diagm((β / γ) * (Rss * Rss / Wss) * (ξ * wweights) .* (ell .^ (-(1.0 + γ) / γ)) .* (ell .^ (-1.0 / γ) .<= wgrid)))

#KF Equation
JJ[F2, LM] = AA' * diagm(wweights)

JJ[F2, LELL] = -(Rss / Wss) * (1.0 / γ) * ζ' * diagm(μ .* wweights .* (ell .^ (-(1.0 + γ) / γ)) .* (ell .^ (-1.0 / γ) .<= wgrid))

JJ[F2, KK] = -(1.0 / Wss) * (ζ' .* repeat((wgrid - c)', nw, 1)) * (μ .* wweights) * dRdK - (μRHS / Wss + (1.0 / (Wss * Wss)) * (ζ' .* (repeat(wgrid, 1, nw) - Rss * repeat((wgrid - c)', nw, 1))) * (μ .* wweights)) * dWdK

JJ[F2, Z] = -(1.0 / Wss) * (ζ' .* repeat((wgrid - c)', nw, 1)) * (μ .* wweights) * dRdZ - (μRHS / Wss + (1.0 / (Wss * Wss)) * (ζ' .* (repeat(wgrid, 1, nw) - Rss * repeat((wgrid - c)', nw, 1))) * (μ .* wweights)) * dWdZ

JJ[F2, M] = -I(nw)

#DEFN of LM(t+1) = M(t)  
JJ[F3, LMP] = I(nw)

JJ[F3, M] = -I(nw)

#DEFN of LELL(t+1) = ELL(t) 
JJ[F4, LELLP] = I(nw)

JJ[F4, ELL] = -I(nw)

# LOM K
JJ[F5, ELL] = -(1.0 / γ) * μ' * (diagm(wweights .* (ell .^ (-(1.0 + γ) / γ)) .* (ell .^ (-1.0 / γ) .<= wgrid)))

JJ[F5, M] = -(wgrid - c)' * diagm(wweights)

JJ[F5, KKP] .= 1.0

# TFP
JJ[F6, ZP] .= 1.0

JJ[F6, Z] .= -rhoz

# # create q matrix which we will use later
qqq = Matrix(I(nw))
qqq[:, 1] = ones(nw)
(qqq1, rrr1) = qr(qqq)

QW = qqq1[:, 2:end]'; #
Qleft = cat(I(nw), QW, QW, I(nw), [1], [1], dims=(1, 2))
Qx = cat(QW, I(nw), [1], [1], dims=(1, 2))
Qy = cat(QW, I(nw), dims=(1, 2))


gx2, hx2, gx, hx = solve(JJ, Qleft, Qx, Qy)


# make IRF's

# from now on we only plot things for cash grid points
# with at least mindens = 1e-8 density in steady state
mindens = 1e-8
maxw = findlast(μ .> mindens)
wg = wgrid[1:maxw]
dur = 50

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

    p9 = plot(wg, c[1:maxw], xlabel="w", title="cSSpolicy", legend=false)
    savefig(p9, "plots/KS/cSSpolicy.png")

    p10 = plot(wg, μ[1:maxw], xlabel="w", title="wealthSSdist", legend=false)
    savefig(p10, "plots/KS/wealthSSdist.png")

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
