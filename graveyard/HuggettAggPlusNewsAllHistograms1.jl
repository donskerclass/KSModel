#HuggettAggPlusNewsAllHistograms1.jl
    #Exactly as HuggettAggPlusNewsAllHistograms.jl, except plot command updated to new syntax needed for current version of PyPlot
using Distributions
using ForwardDiff
# using ChebyshevStuff
using PyPlot

########################################################################
####################### set parameters #################################
########################################################################

bbeta = 0.95          # discount factor
gama  = 2.0           # CRRA Parameter
In    = 0.443993816237631 # normalizing constant for the mollifier
smoother = 1.0        # scale of the mollifier (how peaked the mollifier is) sommther = 0 you get uniform dist
                      # harder to numerically evaluate for small smoother
sigs = 1.0            # sigma of logn in income
nw = 150               # # cash on hand ditn grid points
ns = 10               # # income distribution grid points
alo   = -0.55     # borrowing limit
ahi   = 15.0           # upper bound on asset grid
slo = 0.5             # lower bound on income grid
shi  = 3.5            # upper bound on income
sigs = 1.0            # sigma of log normal in income
sscale = (shi-slo)/2  # size of our s grid, relative to size of [-1,1]
zlo = 0.0             # second income shock to mollify actual income
zhi  = 2.0            # upper bound on this process
smoother = 1.0;       # scale of the mollifier (how peaked the mollifier is) sommther = 0 you get uniform dist
                      # harder to numerically evaluate for small smoother
wlo = alo + slo + zlo # lower bound on cash on hand
whi = ahi + shi + zhi # upper bound on cash on hand
wscale = (whi-wlo)  # size of our w grid, relative to size of [-1,1]
rhoag = 0.9           # persistence of shocks to aggregate component of income
rhosig = 0.5         # persistence of shocks to vol of iid income, known 1 period ahead
agss = 0.5           # mean aggregate income

########################################################################

#wcheb    = chebpts(nw,wlo,whi,2) # wgrid is cash on had grid, wweights are quadrature weights
#wgrid    = wcheb[1]'
#wweights = squeeze(wcheb[2]',2)
# replace Chebyshev grid and weights in w by histograms
wgrid    = collect(linspace(wlo,whi,nw))         #Evenly spaced grid
wweights = (wscale/nw)*ones(nw)        #Equal weights, scaled to width of intervals


scheb    = chebpts(ns,slo,shi,2)
sgrid    = scheb[1]'
sweights = squeeze(scheb[2]',2)

MW = sqrt(wscale/nw)*eye(nw)
# matrix that maps from set of values of a functions at a point to the set of normalized scaling function coefficients
MWinv=sqrt(nw/(wscale))*eye(nw) # pseudoinverse of MW

# make the mollifier function which we will use later
function pointmol(z::AbstractFloat,zhi::AbstractFloat,
                  zlo::AbstractFloat,smoother::AbstractFloat)
    # pointwise mollifier
    In = 0.443993816237631
    if z<zhi && z>zlo
        temp = -1.0 + 2.0*(z-zlo)/(zhi-zlo)
        temp = temp/smoother
        out = ((zhi-zlo)/2.0)*exp(-1.0/(1.0-temp*temp))/(In*smoother);
    else
        out = 0.0
    end
    return out
end

function trunc_lognpdf(x, UP, LO, mu, sig)
    # truncated log normal pdf
    logn = LogNormal(mu, sig)
    return (x-LO.>=-eps()).*(x-UP.<=eps()).*pdf(logn,x)/(cdf(logn,UP)-cdf(logn,LO))
end

g   = trunc_lognpdf(sgrid, shi, slo, 0.0, sigs) # our income distr
#swb = max.(broadcast(-,wgrid,bgrid'), slo/2) # the s you would have if you went from b to w - use slo/2 to avoid -ve numbers
#gwb = trunc_lognpdf(swb, shi, slo, 0.0, sigs)

# make R

Win = ones(nw) # guess for the W function W(w) = beta R E u'(c_{t+1})
# try bisection method
Rlo = 0.4/bbeta # excess should be negative at this value.
Rhi = 1/bbeta # excess is +

function ellfunction(bbeta::AbstractFloat,
                     R::AbstractFloat,
                     gama::AbstractFloat,
                     smoother::AbstractFloat,
                     sigs::AbstractFloat,
                     zhi::AbstractFloat,
                     zlo::AbstractFloat,
                     alo::AbstractFloat,
                     wgrid::Vector{Float64},
                     sgrid::Vector{Float64},
                     wweights::Vector{Float64},
                     sweights::Vector{Float64},
                     Win::Vector{Float64},
                     g::Vector{Float64},
                     agss::AbstractFloat)
    # outputs: [c,ap,Wout,tr]

    nw = length(wgrid)
    ns = length(sgrid)
    c  = zeros(nw,1) # consumption
    ap = zeros(nw,1) # savings
    damp = 0.7 # how much we update W guess
    count = 1
    dist = 1.0 # initial tolerance on decision rules
    tol = 1e-7
    Wout = copy(Win)
    q = zeros(nw,nw*ns)
    tr = zeros(nw,nw)
    while dist>tol && count<5000
        # compute c(w) given guess for Win = bbeta*R*E[u'(c_{t+1})]
        c = min.(Win.^(-1.0/gama),wgrid+agss-alo/R)
        # compute ap(w) given guess for Win
        ap = wgrid + agss - c
        for iw=1:nw
            for iwp=1:nw
                for iss=1:ns
                    q[iw,nw*(iss-1)+iwp] = pointmol(wgrid[iwp] - R*ap[iw] - sgrid[iss],zhi,zlo,smoother)
                end
            end
        end
        Wout = bbeta*R*q*kron(sweights.*g,wweights.*(c.^(-gama)))
        dist = maximum(abs.(Wout-Win))
        Win = damp*Wout + (1.0-damp)*Win;
        count += 1
    end
    if count==5000
        warn("Euler iteration did not converge")
    end
    # tempKF = repmat(wgrid,1,nw*ns) - R*repmat(ap',nw,ns) - kron(sgrid',ones(nw,nw))
    for iw=1:nw
        for iwp=1:nw
            sumns = 0.0
            for isp=1:ns
                sumns += pointmol(wgrid[iwp] - R*ap[iw] - sgrid[isp],zhi,zlo,smoother)*g[isp]*sweights[isp]
            end
            tr[iwp,iw] = sumns
        end
    end
    #tr = q*kron(sweights.*g,eye(nw))
    return (c,ap,Wout,tr)
end

function findss(bbeta::AbstractFloat,
                Rlo::AbstractFloat,
                Rhi::AbstractFloat,
                gama::AbstractFloat,
                smoother::AbstractFloat,
                sigs::AbstractFloat,
                zhi::AbstractFloat,
                zlo::AbstractFloat,
                alo::AbstractFloat,
                wgrid::Vector,
                sgrid::Vector,
                wweights::Vector,
                sweights::Vector,
                Win::Vector,
                g::Vector,
                agss::AbstractFloat)
    excess = 5000.0 # excess supply of savings
    tol = 1e-5
    count=1
    maxit=100
    c = zeros(nw)
    ap = zeros(nw)
    KF = zeros(nw,nw)
    wdist = zeros(nw)
    R = (Rlo+Rhi)/2.0
    D=zeros(size(MW))
    while abs(excess)>tol && count<maxit # clearing markets
        R = (Rlo+Rhi)/2.0
        (c, ap, Win, KF) = ellfunction(bbeta, R, gama, smoother, sigs,
                                       zhi, zlo, alo, wgrid, sgrid,
                                       wweights, sweights, Win, g, agss)
        # ellfunction returns the consumption decision rule, a prime
        # decision rule, Win here is the correct value of
        # beta R u'(c_{t+1}), KF is the Kolmogorov foward operator
        # and is the map which moves you from b distribution
        # today to b dist tomorrow
        # ellfunction is what the pdf calls exp(l(w))

        # Calculate invariant distribution by polynomial interpolation
        LPMKF=MW*KF*MW' #Obtain Legendre polynomial coefficient representation of transition matrix
        # These coefficient are for Legendre polynomial deined on [blo,bhi].
        # no further rescaling required

        # find eigenvalue closest to 1
        (D,V) = eig(LPMKF)
        if abs(D[1]-1)>2e-1 # that's the tolerance we are allowing
            warn("your eigenvalue is too far from 1, something is wrong")
        end
        wdist = MWinv*real(V[:,1]) #Pick the eigen vecor associated with the largest
        # eigenvalue and movesing it back from the coefficients of Legendre
        # polynomials to actual function value on the Chebshev grid

        wdist = wdist/(wweights'*wdist) #Scale of eigenvectors not determinate: rescale to integrate to exactly 1
        excess = wweights'*(wdist.*ap)  #compute excess supply of savings, which is a fn of w

        # bisection
        if excess>0
            Rhi=R
        elseif excess<0
            Rlo = R
        end
        #println(R)
        count += 1
    end
    return (Win, c, wdist, R, D)
end

tic()
(ell, c, m, R, D) = findss(bbeta, Rlo, Rhi, gama, smoother, sigs,
                            zhi, zlo, alo, wgrid, sgrid,
                            wweights, sweights, Win, g, agss)
toc()

# Calculate orthogonalization wrt constant vector to ensure probabilities integrate to 1

qqq=zeros(nw,nw)
qqq[:,1]=ones(nw)
qqq[:,2:end]=MW[:,2:end]
(qqq1,rrr1)=qr(qqq)


# set up equilibrium conditions

function molly(x)
    In = 0.443993816237631
    temp = (-1.0 + 2.0*(x-zlo)/(zhi-zlo))/smoother
    y  = ((zhi-zlo)/2.0)*exp.(min.(-1.0./(1.0 - temp.^2),2.0)    )
    y  = (y/(In*smoother))
    y  =  y.*(x.>zlo).*(x.<zhi)
end

gfun(lsig)   = trunc_lognpdf(sgrid,shi,slo,(1.0-exp.(2*lsig))/2.0,exp.(lsig))

#Euler Equation
cfunc(ell,R,ag) = min.(ell.^(-1.0/gama),wgrid + ag - alo/R)
mollificand(ell,R,ag) = repmat(wgrid',nw,ns) - R*repmat(wgrid+ag-cfunc(ell,R,ag),1,nw*ns) - kron(sgrid',ones(nw,nw))
fancyF1(ell, ellp, ag, agp, R, Rp, lsig) = ell - bbeta*R*molly(mollificand(ell,R,ag))*kron(sweights.*gfun(lsig),wweights.*(cfunc(ellp,Rp,agp).^(-gama)))

#KF equation
KFmollificand(ell,R,ag) = repmat(wgrid,1,nw*ns) - R*repmat(wgrid'+ag-cfunc(ell,R,ag)',nw,ns) - kron(sgrid',ones(nw,nw))
fancyF2(mp, m, ell, R, ag, lsig) = mp - molly(KFmollificand(ell,R,ag))*kron(sweights.*gfun(lsig),wweights.*m)

#transition of ag
fancyF3(ag,agp) = agp-rhoag*ag

# transition of lsig
fancyF4(lsig, lsigp) = lsigp - rhosig*lsig

#mkt clearing
fancyF5(ell,R,m,ag) = sum((wgrid + ag - cfunc(ell,R,ag)).*(m.*wweights))


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
JFD = x->ForwardDiff.jacobian(Fstack,x)
tic()
Jac0 = JFD(usex)
toc()

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
InvPimats = [diagm(1./wweights),1];
outstate  = [1 1 2 2 2];
instate   = [1 2 2 1 2 1 2 2 1 2];
QW        = qqq1[:,2:end]'; #
Qleft     = cat([1,2],eye(nw),QW,[1],[1],[1])
Qx        = cat([1 2],QW,[1],[1]);
Qy        = cat([1 2],eye(nw),[1]);
C2Vx      = cat([1 2],MWinv,[1],[1]);
C2Vy      = cat([1 2],MWinv,[1]);

tic()
include("solve.jl")
toc()

gx2, hx2, gx, hx =  solve(Jac0, funops, FF, vars, outinds, Mmats,
                              InvPimats, outstate, instate, Qleft, Qx,
                              Qy, C2Vy, C2Vx)


## Compute IRFs to shocks!

close("all")


# from now on we only plot things for cash grid points
# with at least mindens = 1e-8 density in steady state
mindens = 1e-8
maxw = find(m.>mindens)[end]
wg = wgrid[1:maxw]
dur =10 #Periods to compute IRFs

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

thingtoplot = agIRFm[1:maxw,:];
fig = figure(figsize=(8,6))
#ax = fig[:gca](projection="3d")
ax = Axes3D(fig)
ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
xgrid = repmat(wg,1,dur);
ygrid = repmat((1:dur)',maxw,1);
ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
                  edgecolors="k",linewidth=0.25)
xlabel("w")
ylabel("t")
zlabel("m_t(w)")
title("cash distribution response to aggregate income shock")


thingtoplot = agIRFc[1:maxw,:];
fig = figure(figsize=(8,6))
#ax = fig[:gca](projection="3d")
ax = Axes3D(fig)
ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
xgrid = repmat(wg,1,dur);
ygrid = repmat((1:dur)',maxw,1);
ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
                  edgecolors="k",linewidth=0.25)
xlabel("w")
ylabel("t")
zlabel("c_t(w)")
title("consumption response to aggregate income shock")

figr=figure()
plot(1:dur,agIRFr)
xlabel("t")
ylabel("R_t")
title("interest rate response to aggregate income shock")

# mu irf
figr=figure()
plot(1:dur,agIRFag)
xlabel("t")
ylabel("ag_t")
title("aggregate income shock")

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
fig = figure(figsize=(8,6))
#ax = fig[:gca](projection="3d")
ax = Axes3D(fig)
ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
xgrid = repmat(wg,1,dur);
ygrid = repmat((1:dur)',maxw,1);
ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
                  edgecolors="k",linewidth=0.25)
xlabel("w")
ylabel("t")
zlabel("m_t(w)")
title("cash distribution response to risk shock")


thingtoplot = lsIRFc[1:maxw,:];
fig = figure(figsize=(8,6))
#ax = fig[:gca](projection="3d")
ax = Axes3D(fig)
ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
xgrid = repmat(wg,1,dur);
ygrid = repmat((1:dur)',maxw,1);
ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
                  edgecolors="k",linewidth=0.25)
xlabel("w")
ylabel("t")
zlabel("c_t(w)")
title("consumption response to risk shock")

figr=figure()
plot(1:dur,lsIRFr)
xlabel("t")
ylabel("R_t")
title("interest rate response to risk shock")

# mu irf
figr=figure()
plot(1:dur,lsIRFls)
xlabel("t")
ylabel("lsig_t")
title("risk shock")

# still to do: plots of consumption distribution and other auxiliary variables
# in particular g, after sig shock

# mean consumption
meanc(c,m) = sum(m.*wweights.*c)
varc(c,m)  = sum(m.*wweights.*((c-meanc(c,m)).^2))
meanw(m)   = sum(m.*wweights.*wgrid) # should be 0
varw(m)    = sum(m.*wweights.*((wgrid-meanw(m)).^2))

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

#close("all")

fig=figure()
plot(1:dur,agIRFmc',1:dur,10*agIRFvc',1:dur,agIRFmw',1:dur,10*agIRFvw')
legend(("mean c", "10*var c", "mean cash", "10*var cash"))
xlabel("t")
title("shock to aggregate income")

fig=figure()
plot(1:dur,lsIRFmc',1:dur,10*lsIRFvc',1:dur,lsIRFmw',1:dur,10*lsIRFvw')
legend(("mean c", "10*var c", "mean cash", "10*var cash"))
xlabel("t")
title("shock to log variance of idiosyncratic income")

fig=figure()
subplot(121)
plot(wgrid,c)
xlabel("Cash on hand")
ylabel("Consumption")
title("Steady State Policy Function")
subplot(122)
plot(wgrid,m)
xlabel("Cash on hand")
ylabel("Density")
title("Steady State Wealth Distribution")
