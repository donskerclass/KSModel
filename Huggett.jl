using KSModel, Parameters, LinearAlgebra, Plots, ForwardDiff
huggett_params = huggett_model()
wgrid, wweights, scheb, sgrid, sweights, MW, MWinv = grids(huggett_params)
@unpack rhoag, rhosig, nw, ns, sigs, shi, slo, alo, bbeta, zhi, zlo, smoother, gama, agss = huggett_params
g = trunc_lognpdf.(sgrid, Ref(shi), Ref(slo), Ref(0.0), Ref(sigs)) # density g evaluated on sgrid

Win = ones(nw) # guess for the W function W(w) = beta R E u'(c_{t+1})
Rlo = 0.4/bbeta # excess should be negative at this value.
Rhi = 1/bbeta # excess is +

@time (ell, c, m, R, D) = huggett_findss(huggett_params, Rlo, Rhi, wgrid, sgrid, wweights, sweights, Win, g, MW, MWinv)

# Calculate orthogonalization wrt constant vector to ensure probabilities integrate to 1
qqq=zeros(nw,nw) # TODO: should this be ones? see L141 KS.jl
qqq[:,1]=ones(nw)
qqq[:,2:end]=MW[:,2:end]
(qqq1,rrr1)=qr(qqq)


# set up equilibrium conditions
function molly(x)
    In = 0.443993816237631
    temp = (-1.0 .+ 2.0 * (x .- zlo) / (zhi .- zlo)) / smoother
    y = ((zhi - zlo) / 2.0) * exp.(min.(-1.0 ./ (1.0 .- temp .^ 2), 2.0))
    y = (y / (In * smoother))
    y = y .* (x .> zlo) .* (x .< zhi)
end

gfun(lsig) = trunc_lognpdf.(sgrid, Ref(shi), Ref(slo), Ref((1.0 - exp.(2*lsig))/2.0), Ref(exp.(lsig)))
cfunc(ell, R, ag) = min.(ell.^(-1.0/gama), wgrid .+ ag .- alo/R)

# Euler equation
mollificand(ell,R,ag) = repeat(wgrid', nw, ns) - R * repeat(wgrid .+ ag .- cfunc(ell, R, ag),1,nw*ns) - kron(sgrid',ones(nw, nw))
fancyF1(ell, ellp, ag, agp, R, Rp, lsig) = ell - bbeta*R*molly(mollificand(ell,R,ag))*kron(sweights.*gfun(lsig),wweights.*(cfunc(ellp,Rp,agp).^(-gama)))

# KF Equation
KFmollificand(ell,R,ag) = repeat(wgrid,1,nw*ns) - R*repeat(wgrid'.+ag.-cfunc(ell,R,ag)',nw,ns) - kron(sgrid',ones(nw,nw))
fancyF2(mp, m, ell, R, ag, lsig) = mp - molly(KFmollificand(ell,R,ag))*kron(sweights.*gfun(lsig),wweights.*m)

fancyF3(ag,agp) = agp-rhoag*ag # ag transition
fancyF4(lsig, lsigp) = lsigp - rhosig*lsig # lsig transition
fancyF5(ell,R,m,ag) = sum((wgrid .+ ag .- cfunc(ell,R,ag)).*(m.*wweights)) # market clearing 

# stack all equilibrium conditions
F(mp,agp,lsigp,ellp,Rp,m,ag,lsig,ell,R) =
                                [fancyF1(ell, ellp, ag, agp, R, Rp, lsig);
                                fancyF2(mp, m, ell, R, ag, lsig);
                                fancyF3(ag,agp);
                                fancyF4(lsig,lsigp);
                                fancyF5(ell,R,m,ag)
                                ]

# we will always order things XP YP X Y
# convention is that capital letters generally refer to indices
MP    = 1     : nw
AGP   = nw+1
LSIGP = nw+2
ELLP  = nw+3  :2*nw+2
RRP   = 2*nw+3
M     = 2*nw+4:3*nw+3
AG    = 3*nw+4
LSIG  = 3*nw+5
ELL   = 3*nw+6:4*nw+5
RR    = 4*nw+6

Fstack(x) = F(x[MP],x[AGP],x[LSIGP],x[ELLP],x[RRP],x[M],x[AG],x[LSIG],x[ELL],x[RR])

# make jacobian
usex = [m;agss;log(sigs);ell;R;m;agss;log(sigs);ell;R]
@time JFD = x->ForwardDiff.jacobian(Fstack,x)
Jac0 = JFD(usex)


# create objects needed for solve.jl
funops = 1:2 # which operators output a function?
F1 = 1     :nw
F2 = nw+1  :2*nw
F3 = 2*nw+1:2*nw+1
F4 = 2*nw+2:2*nw+2
F5 = 2*nw+3:2*nw+3
FF      = [F1,F2,F3,F4,F5]                          # indices of operators
vars    = [MP,AGP,LSIGP,ELLP,RRP,M,AG,LSIG,ELL,RR] # indices of variables
outinds = [9,1,0,0,0];                             # indices of output args
Mmats     = [MW,1];
InvPimats = [diagm(1 ./ wweights),1];
outstate  = [1 1 2 2 2];
instate   = [1 2 2 1 2 1 2 2 1 2];
QW        = qqq1[:,2:end]'; #
Qleft     = cat(I(nw),QW,[1],[1],[1], dims = (1, 2))
Qx        = cat(QW,[1],[1], dims = (1, 2));
Qy        = cat(I(nw),[1], dims = (1, 2));
C2Vx      = cat(MWinv,[1],[1], dims = (1, 2));
C2Vy      = cat(MWinv,[1], dims = (1, 2));

gx2, hx2, gx, hx =  huggett_solve(Jac0, funops, FF, vars, outinds, Mmats,
                                InvPimats, outstate, instate, Qleft, Qx,
                                Qy, C2Vy, C2Vx)


# ## Compute IRFs to shocks!

# close("all")


# from now on we only plot things for cash grid points
# with at least mindens = 1e-8 density in steady state
mindens = 1e-8
maxw = findlast(m .> mindens)
wg = wgrid[1:maxw]
dur = 10 #Periods to compute IRFs

# we will need this to make consumption shock
cfstack(xx) = cfunc(xx[1:nw],xx[nw+1],xx[nw+2]);
dcdx    = xx->ForwardDiff.jacobian(cfstack,xx);
usexx=[ell;R;agss];
dcdxss  = dcdx(usexx);

# first, shock to aggregate component of income
agshock = 1.0
agIRFxc = zeros(nw+1,dur) #IRF to income shock, coefficient values
agIRFxc[nw,1] =agshock
for t=1:dur-1
    agIRFxc[:,t+1] = hx*agIRFxc[:,t]
end
agIRFyc = gx*agIRFxc #y response

#Shocks in terms of grid values
agIRFxp = C2Vx*Qx'agIRFxc
agIRFyp = C2Vy*Qy'agIRFyc

agIRFell = agIRFyp[1:nw,:]
agIRFr   = agIRFyp[nw+1,:]
agIRFm   = agIRFxp[1:nw,:]
agIRFag  = agIRFxp[nw+1,:]
# sig shock is obviously zero so we omit it
# make consumption shock:
agIRFc = dcdxss*[agIRFell;agIRFr';agIRFag'];

pyplot()

thingtoplot = agIRFm[1:maxw,:];
xgrid = repeat(wg,1,dur);
ygrid = repeat((1:dur)',maxw,1);
plot(xgrid, ygrid, thingtoplot, st = :surface, xlabel = "w", ylabel = "t", zlabel = "m_t(w)", title = "cash dist. response to aggregate income shock")

thingtoplot = agIRFc[1:maxw,:];
xgrid = repeat(wg,1,dur);
ygrid = repeat((1:dur)',maxw,1);
plot(xgrid, ygrid, thingtoplot, st = :surface, xlabel = "w", ylabel = "t", zlabel = "c_t(w)", title = "consumption response to aggregate income shock")

plot(1:dur,agIRFr, xlabel = "t", ylabel = "R_t", title="interest rate response to aggregate income shock", legend = false)

plot(1:dur,agIRFag, xlabel = "t", ylabel = "ag_t", title="aggregate income shock", legend = false)


# next, risk shock
lsshock = 1.0
lsIRFxc = zeros(nw+1,dur) #IRF to lsig shock, coefficient values
lsIRFxc[nw+1,1] =lsshock
for t=1:dur-1
    lsIRFxc[:,t+1] = hx*lsIRFxc[:,t]
end
lsIRFyc = gx*lsIRFxc #y response

#Shocks in terms of grid values
lsIRFxp = C2Vx*Qx'lsIRFxc
lsIRFyp = C2Vy*Qy'lsIRFyc

lsIRFell = lsIRFyp[1:nw,:]
lsIRFr   = lsIRFyp[nw+1,:]
lsIRFm   = lsIRFxp[1:nw,:]
lsIRFag  = lsIRFxp[nw+1,:] # this should be 0
lsIRFls  = lsIRFxp[nw+2,:] # this should be 0
# make consumption shock:
lsIRFc = dcdxss*[lsIRFell;lsIRFr';lsIRFag'];

thingtoplot = lsIRFm[1:maxw,:];
xgrid = repeat(wg,1,dur);
ygrid = repeat((1:dur)',maxw,1);
plot(xgrid, ygrid, thingtoplot, st = :surface, xlabel = "w", ylabel = "t", zlabel = "m_t(w)", title = "cash dist. response to risk shock")

thingtoplot = lsIRFc[1:maxw,:];
xgrid = repeat(wg,1,dur);
ygrid = repeat((1:dur)',maxw,1);
plot(xgrid, ygrid, thingtoplot, st = :surface, xlabel = "w", ylabel = "t", zlabel = "c_t(w)", title = "consumption response to risk shock")

plot(1:dur,lsIRFr, xlabel = "t", ylabel = "R_t", title = "interest rate response to risk shock", legend = false)

plot(1:dur,lsIRFls,xlabel="t",ylabel="lsig_t",title="risk shock")

# still to do: plots of consumption distribution and other auxiliary variables
# in particular g, after sig shock

# mean consumption
meanc(c,m) = sum(m.*wweights.*c)
varc(c,m)  = sum(m.*wweights.*((c.-meanc(c,m)).^2))
meanw(m)   = sum(m.*wweights.*wgrid) # should be 0
varw(m)    = sum(m.*wweights.*((wgrid.-meanw(m)).^2))

mcstack(xx) = meanc(xx[1:nw],xx[nw+1:2*nw])
dmcdx   = xx->ForwardDiff.gradient(mcstack,xx)
usexx   = [c;m]
dmcdxss = dmcdx(usexx)
agIRFmc = dmcdxss'*[agIRFc;agIRFm]
lsIRFmc = dmcdxss'*[lsIRFc;lsIRFm]

vcstack(xx) = varc(xx[1:nw],xx[nw+1:2*nw])
dvcdx   = xx->ForwardDiff.gradient(vcstack,xx)
dvcdxss = dvcdx(usexx)
agIRFvc = dvcdxss'*[agIRFc;agIRFm]
lsIRFvc = dvcdxss'*[lsIRFc;lsIRFm]

dmwdm   = m->ForwardDiff.gradient(meanw,m)
dmwdmss = dmwdm(m)
agIRFmw = dmwdmss'*agIRFm
lsIRFmw = dmwdmss'*lsIRFm

dvwdm   = m->ForwardDiff.gradient(varw,m)
dvwdmss = dvwdm(m)
agIRFvw = dvwdmss'*agIRFm
lsIRFvw = dvwdmss'*lsIRFm

# #close("all")

# fig=figure()
# plot(1:dur,agIRFmc',1:dur,10*agIRFvc',1:dur,agIRFmw',1:dur,10*agIRFvw')
# legend(("mean c", "10*var c", "mean cash", "10*var cash"))
# xlabel("t")
# title("shock to aggregate income")

# fig=figure()
# plot(1:dur,lsIRFmc',1:dur,10*lsIRFvc',1:dur,lsIRFmw',1:dur,10*lsIRFvw')
# legend(("mean c", "10*var c", "mean cash", "10*var cash"))
# xlabel("t")
# title("shock to log variance of idiosyncratic income")

# fig=figure()
# subplot(121)
# plot(wgrid,c)
# xlabel("Cash on hand")
# ylabel("Consumption")
# title("Steady State Policy Function")
# subplot(122)
# plot(wgrid,m)
# xlabel("Cash on hand")
# ylabel("Density")
# title("Steady State Wealth Distribution")
