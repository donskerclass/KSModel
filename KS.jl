using Distributions
using ForwardDiff 
using ChebyshevStuff
using PyPlot


# set parameters 
tic()
β = 0.95                        # discount factor
γ  = 3.0                        # CRRA Parameter
α = 1.0/3.0                     # capital share
δ  = 0.2                        # depreciation
In    = 0.443993816237631       # normalizing constant for the mollifier
smoother = 1.0                  # scale of the mollifier (how peaked the mollifier is) smoother = 0 you get uniform dist
nw = 160                         # cash on hand ditn grid points
ns = 50                         # skill distribution grid points
slo = 0.5                       # lower bound on skill grid
shi = 1.5                       # upper bound on skill
sigs = 0.5                      # sigma of log normal in income
sscale = (shi-slo)/2.0          # size of our s grid, relative to size of [-1,1]
zlo =  0.0                      # second income shock to mollify actual income
zhi = 2.0                       # upper bound on this process
wlo = 0.5                       # lower bound on cash on hand
whi = 15.0                      # upper bound on cash on hand
wscale = (whi-wlo)              # size of w grids
rhoz = 0.95                     # persistence of agg tfp

# make grids
wgrid    = collect(linspace(wlo,whi,nw)) #Evenly spaced grid
wweights = (wscale/nw)*ones(nw)          #quadrature weights

scheb    = chebpts(ns,slo,shi,2)          
sgrid    = scheb[1]'                     # sgrid is grid for "skill"
sweights = squeeze(scheb[2]',2)          # Curtis-Clenshaw quadrature weights

MW = sqrt(wscale/nw)*eye(nw)             # matrix that maps from set of values of a functions at a point to the set of normalized scaling function coefficients

MWinv=sqrt(nw/(wscale))*eye(nw)          # inverse of MW


function mollifier(z::AbstractFloat,zhi::AbstractFloat,
                  zlo::AbstractFloat,smoother::AbstractFloat)
    # mollifier function
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
    return (x-LO.>=-eps()).*(x-UP.<=eps()).*pdf.(logn,x)/(cdf(logn,UP)-cdf(logn,LO))
end

g = trunc_lognpdf(sgrid, shi, slo, 0.0, sigs) # density g evaluated on sgrid
L = sum(sweights.*g.*sgrid)                   # Aggregate Labor endowment

MPK(Z::AbstractFloat,K::AbstractFloat) = α*Z*(K^(α-1.0))*(L^(1.0-α)) + 1.0-δ   # Marginal product of capital + 1-δ
MPL(Z::AbstractFloat,K::AbstractFloat) = (1.0-α)*Z*(K^α)*(L^(-α))  # Marginal product of labor

# make K
K = 1.5
#L*(α/((1.0/β)-1.0+δ))^(1.0/(1.0-α))

# guess for the W function W(w) = beta R E u'(c_{t+1})
Win = 0.3*ones(nw) 


function sspolicy(β::AbstractFloat,
                     K::AbstractFloat,
                     γ::AbstractFloat,
                     α::AbstractFloat,
                     δ::AbstractFloat,
                     L::AbstractFloat,
                     smoother::AbstractFloat,
                     sigs::AbstractFloat,
                     zhi::AbstractFloat,
                     zlo::AbstractFloat,
                     wgrid::Vector{Float64},
                     sgrid::Vector{Float64},
                     wweights::Vector{Float64},
                     sweights::Vector{Float64},
                     Win::Vector{Float64},
                     g::Vector{Float64})
    # outputs: [c,ap,Wout,tr]

    nw = length(wgrid)
    ns = length(sgrid)
    c  = zeros(nw)      # consumption
    ap = zeros(nw)      # savings
    damp = 0.8          # how much we update W guess
    count = 1
    dist = 1.0          # distance between Win and Wout on decision rules
    tol = 1e-5          # acceptable error 
    Wout = copy(Win)    # initialize Wout
    R = α*(K^(α-1.0))*(L^(1.0-α)) + 1.0-δ 
    wage = (1.0-α)*(K^α)*(L^(-α))
    tr = zeros(nw,nw)   # transition matrix mapping todays dist of w to w'
    q = zeros(nw,nw*ns) # auxilary variable to compute LHS of euler
    while dist>tol && count<5000
        # compute c(w) given guess for Win = β*R*E[u'(c_{t+1})]
        c = min.(Win.^(-1.0/γ),wgrid)
        ap = wgrid - c  # compute ap(w) given guess for Win
        for iw=1:nw
            for iwp=1:nw
                for iss=1:ns
                    q[iw,nw*(iss-1)+iwp] = mollifier((wgrid[iwp] - R*ap[iw])/wage - sgrid[iss],zhi,zlo,smoother)
                end
            end
        end
        Wout = β*R*q*kron(sweights.*(g/wage),wweights.*(c.^(-γ)))
        dist = maximum(abs.(Wout-Win))
        Win = damp*Wout + (1.0-damp)*Win;
        count += 1
    end
    if count==5000
        warn("Euler iteration did not converge")
    end

    for iw=1:nw
        for iwp=1:nw
            sumns = 0.0
            for isp=1:ns
                sumns += mollifier((wgrid[iwp] - R*ap[iw])/wage - sgrid[isp],zhi,zlo,smoother)*(g[isp]/wage)*sweights[isp]
            end
            tr[iwp,iw] = sumns
        end
    end
    return (c,ap,Wout,tr)
end

function findss(β::AbstractFloat,
                K::AbstractFloat,
                γ::AbstractFloat,
                α::AbstractFloat,
                δ::AbstractFloat,
                L::AbstractFloat,
                smoother::AbstractFloat,
                sigs::AbstractFloat,
                zhi::AbstractFloat,
                zlo::AbstractFloat,
                wgrid::Vector,
                sgrid::Vector,
                wweights::Vector,
                sweights::Vector,
                Win::Vector{Float64},
                g::Vector{Float64})
    tol = 1e-5
    count=1
    maxit=100
    c = zeros(nw)
    ap = zeros(nw)
    KF = zeros(nw,nw)
    wdist = zeros(nw)
    diseqm = 1234.5
    newK = 0.0
    Kdamp = 0.1
    while abs(diseqm)>tol && count<maxit # clearing markets
        (c, ap, Win, KF) = sspolicy(β, K, γ, α, δ, L, smoother, sigs,
                                       zhi, zlo, wgrid, sgrid,
                                       wweights, sweights, Win, g)
        # sspolicy returns the consumption decision rule, a prime
        # decision rule, Win = updated β R u'(c_{t+1}), 
        # KF is the Kolmogorov foward operator
        # and is the map which moves you from cash on hand distribution
        # today to cash on had dist tomorrow

        LPMKF=MW*KF*MW'
        
        # find eigenvalue closest to 1
        (D,V) = eig(LPMKF)
        if abs(D[1]-1)>2e-1 # that's the tolerance we are allowing
            warn("your eigenvalue is too far from 1, something is wrong")
        end
        wdist = MWinv*real(V[:,1]) #Pick the eigen vecor associated with the largest
        # eigenvalue and moving it back to values

        wdist = wdist/(wweights'*wdist) #Scale of eigenvectors not determinate: rescale to integrate to exactly 1
        newK  = wweights'*(wdist.*ap)  #compute excess supply of savings, which is a fn of w
        diseqm = newK - K
        #println(newK)
        K += Kdamp*diseqm
        count += 1
    end
    return (Win, c, wdist, newK, KF)
end

tic()
(ell, c, μ, K, KF) = findss(β, K, γ, α, δ, L, smoother, sigs,
                            zhi, zlo, wgrid, sgrid,
                            wweights, sweights, Win, g)
toc()

function dmollifier(x::AbstractFloat, zhi::AbstractFloat, zlo::AbstractFloat, smoother::AbstractFloat)
    In = 0.443993816237631
    temp = (-1.0 + 2.0*(x-zlo)/(zhi-zlo))/smoother
    dy  = -(2*temp./((1 - temp.^2).^2)).*(2/(smoother*(zhi-zlo))).*mollifier(x, zhi, zlo, smoother)
end

# Make the Jacobian 
JJ = zeros(4*nw+2,8*nw+4)

# we will always order things XP YP X Y
# convention is that capital letters generally refer to indices
LMP   = 1     :nw
LELLP = nw+1  :2*nw
KKP   = 2*nw+1
ZP    = 2*nw+2
MP    = 2*nw+3:3*nw+2
ELLP  = 3*nw+3:4*nw+2
LM    = 4*nw+3:5*nw+2
LELL  = 5*nw+3:6*nw+2
KK    = 6*nw+3
Z     = 6*nw+4
M     = 6*nw+5:7*nw+4
ELL   = 7*nw+5:8*nw+4

# create objects needed for solve.jl
funops = 1:4 # which operators output a function
F1 = 1     :nw
F2 = nw+1  :2*nw
F3 = 2*nw+1:3*nw
F4 = 3*nw+1:4*nw
F5 = 4*nw+1:4*nw+1
F6 = 4*nw+2:4*nw+2


#Create auxiliary variables
c = min.(ell.^(-1.0/γ),wgrid) # Consumption Decision Rule
Wss = MPL(1.0,K)              # Steady state wages
Rss = MPK(1.0,K)              # Steady state Rk

tic()
#KF equation
KFmollificand = (repmat(wgrid,1,nw*ns) - Rss*repmat(wgrid'-c',nw,ns))/Wss - kron(sgrid',ones(nw,nw))
μRHS = mollifier.(KFmollificand,zhi,zlo,smoother)*kron(sweights.*(g/Wss),wweights.*μ)

ξ = zeros(nw,nw) 
Γ = zeros(nw,nw) 
AA = zeros(nw,nw) 
ζ = zeros(nw,nw) 
for iw=1:nw
    for iwp=1:nw
        sumnsξ=0.0
        sumnsΓ=0.0
        sumnsAA=0.0
        sumnsζ=0.0
        for iss=1:ns
            mf = (wgrid[iwp]-Rss*(wgrid[iw]-c[iw]))/Wss - sgrid[iss]
            dm = dmollifier(mf, zhi, zlo, smoother)
            mm = mollifier(mf, zhi, zlo, smoother)
            sumnsξ += ((c[iwp].^(-γ))/Wss)*dm*g[iss]*sweights[iss]
            sumnsΓ += ((c[iwp].^(-γ))/Wss)*mm*g[iss]*sweights[iss]
            sumnsAA+= (1.0/Wss)*mm*g[iss]*sweights[iss]
            sumnsζ += (1.0/Wss)*dm*g[iss]*sweights[iss]
        end
        ξ[iw,iwp] = sumnsξ
        Γ[iw,iwp] = sumnsΓ
        AA[iw,iwp] = sumnsAA
        ζ[iw,iwp] = sumnsζ
    end
end


dRdK = -(1-α)*((Rss+δ-1)/K)
dWdK = (α*Wss/K)
dRdZ = Rss+δ-1.0
dWdZ = Wss
dYdK = α*(K^(α-1))*(L^(1-α))
dYdZ = (K^α)*(L^(1-α))
ellRHS = β*Rss*Γ*wweights

# Fill in Jacobian

    #Euler equation

JJ[F1,KKP]  = (β*Γ*wweights - β*(Rss/Wss)*((wgrid-c).*(ξ*wweights)))*dRdK - (1.0/Wss)*(ellRHS+((β*Rss)/Wss)*ξ.*(repmat(wgrid',nw,1) - Rss*repmat(wgrid-c,1,nw))*wweights)*dWdK

JJ[F1,ZP]   = (β*Γ*wweights - β*(Rss/Wss)*((wgrid-c).*(ξ*wweights)))*dRdZ - (1.0/Wss)*(ellRHS+((β*Rss)/Wss)*ξ.*(repmat(wgrid',nw,1) - Rss*repmat(wgrid-c,1,nw))*wweights)*dWdZ

JJ[F1,ELLP] = β*Rss*Γ*diagm((wweights.*(ell.^(-(1.0+γ)/γ)).*(ell.^(-1.0/γ) .<= wgrid))./c)

JJ[F1,ELL]  = -( eye(nw) + diagm((β/γ)*(Rss*Rss/Wss)*(ξ*wweights).*(ell.^(-(1.0+γ)/γ)).*(ell.^(-1.0/γ) .<= wgrid)) )

    #KF Equation
JJ[F2,LM]   = AA'*diagm(wweights)

JJ[F2,LELL] = -(Rss/Wss)*(1.0/γ)*ζ'*diagm( μ.*wweights.*(ell.^(-(1.0+γ)/γ)).*(ell.^(-1.0/γ) .<= wgrid))  

JJ[F2,KK]   = -(1.0/Wss)*(ζ'.*repmat( (wgrid-c)',nw,1))*(μ.*wweights)*dRdK - (μRHS/Wss + (1.0/(Wss*Wss))*(ζ'.*(repmat(wgrid,1,nw) - Rss*repmat( (wgrid-c)',nw,1)))*(μ.*wweights))*dWdK

JJ[F2,Z]    = -(1.0/Wss)*(ζ'.*repmat( (wgrid-c)',nw,1))*(μ.*wweights)*dRdZ - (μRHS/Wss + (1.0/(Wss*Wss))*(ζ'.*(repmat(wgrid,1,nw) - Rss*repmat( (wgrid-c)',nw,1)))*(μ.*wweights))*dWdZ

JJ[F2,M]    = -eye(nw)

    #DEFN of LM(t+1) = M(t)  
JJ[F3,LMP]  = eye(nw)

JJ[F3,M]    = -eye(nw)

    #DEFN of LELL(t+1) = ELL(t) 
JJ[F4,LELLP]= eye(nw)

JJ[F4,ELL]  = -eye(nw)

    # LOM K
JJ[F5,ELL]  = -(1.0/γ)*μ'*(diagm(wweights.*(ell.^(-(1.0+γ)/γ)).*(ell.^(-1.0/γ) .<= wgrid)))

JJ[F5,M]    = -(wgrid-c)'*diagm(wweights)

JJ[F5,KKP]  = 1.0

    # TFP
JJ[F6,ZP]   = 1.0

JJ[F6,Z]    = -rhoz

toc()

# create q matrix which we will use later
qqq=eye(nw)
qqq[:,1]=ones(nw)
(qqq1,rrr1)=qr(qqq)

QW        = qqq1[:,2:end]'; #
Qleft     = cat([1 2],eye(nw),QW,QW,eye(nw),[1],[1])
Qx        = cat([1 2],QW,eye(nw),[1],[1])
Qy        = cat([1 2],QW,eye(nw))


include("solve.jl")

tic()
gx2, hx2, gx, hx =  solve(JJ, Qleft, Qx, Qy)
toc()


# make IRF's

close("all")

makefigs=1
if makefigs==1
    # from now on we only plot things for cash grid points
    # with at least mindens = 1e-8 density in steady state
    mindens = 1e-8
    maxw = find(μ.>mindens)[end]
    wg = wgrid[1:maxw]
    dur =50 #Periods to compute IRFs

    # we will need this to make consumption shock
    dcdell = diagm((-1/γ)*(ell.^(-(1/γ)-1)).*((ell.^(-1/γ)).<=wgrid))
    mpc    = [1;(c[2:end] - c[1:end-1])./(wgrid[2:end] - wgrid[1:end-1])]

    # first, shock to aggregate component of income
    zshock        = 0.01
    zIRFxc        = zeros(2*nw+1,dur) #IRF to income shock, coefficient values
    zIRFxc[2*nw+1,1] = zshock
    for t=1:dur-1
        zIRFxc[:,t+1] = hx*zIRFxc[:,t]
    end
    zIRFyc = gx*zIRFxc #y response

    #Shocks in terms of grid values
    zIRFxp  = Qx'zIRFxc
    zIRFyp = Qy'zIRFyc

    zIRFμ    = zIRFyp[1:nw,:]
    zIRFell  = zIRFyp[nw+1:2*nw,:]
    zIRFk    = zIRFxp[2*nw+1,:]
    zIRFz    = zIRFxp[2*nw+2,:]
    zIRFwags = dWdK*zIRFk+dWdZ*zIRFz
    zIRFrags = dRdK*zIRFk+dRdZ*zIRFz
    zIRFgdp  = dYdK*zIRFk+dYdZ*zIRFz
    zIRFcfun = dcdell*zIRFell
    zIRFC    = zIRFcfun'*(wweights.*μ) + zIRFμ'*(wweights.*c)
    zIRFI    = zIRFk[2:end] - (1-δ)*zIRFk[1:end-1]

    yss      = (K^α)*(L^(1-α))
    css      = μ'*(wweights.*c)
    iss      = δ*K

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur,(zIRFgdp/yss)*100)
    xlabel("t")
    savefig("gdpdIRF.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur,(zIRFC/css)*100)
    xlabel("t")
    savefig("aggCIRF.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur-1,(zIRFI/iss)*100)
    xlabel("t")
    savefig("inventoriesIRF.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur,(zIRFz)*100)
    xlabel("t")
    savefig("zIRF.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur,(zIRFrags/Rss)*100)
    xlabel("t")
    savefig("ragsIRF.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(1:dur,(zIRFwags/Wss)*100)
    xlabel("t")
    savefig("wagsIRF.eps")

    thingtoplot = zIRFμ[1:maxw,:];
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
    savefig("wealthIRFdist.eps")


    thingtoplot = zIRFcfun[1:maxw,:];
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
    savefig("cfunIRF.eps")


    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(wg,c[1:maxw])
    xlabel("w")
    savefig("cSSpolicy.eps")

    figure()
    ax=axes()
    ax[:tick_params]("both",labelsize=18)
    plot(wg,μ[1:maxw])
    xlabel("w")
    savefig("wealthSSdist.eps")

    indic = ones(nw,nw)
    indic = indic.*(broadcast(-,1:nw,(1:nw)').>0)

    shock =  1
    μsim  = repmat(μ,1,dur) + shock*zIRFμ
    csim  = repmat(c,1,dur) + shock*zIRFcfun
    Msim  = indic*diagm(wweights)*μsim
    Mss   = indic*diagm(wweights)*repmat(μ,1,dur)

    cstep = 0.04
    clo = [1:cstep:2...]
    chi = clo+cstep
    #clo[1] = c[1]-eps()
    #chi[end] = c[end]
    cgrid = (chi+clo)/2.0
    nbins = length(clo)
    n10c  = zeros(nbins,dur)
    n10s  = zeros(nbins,dur)

    for t=1:dur
        for i=1:nbins
            imin = find(c.>clo[i])[1]
            imax = find(c.<=chi[i])[end]
            n10s[i,:] = Mss[imax,:]-Mss[imin,:]
            n10c[i,:] = Msim[imax,:]-Msim[imin,:]
        end
    end
    qIRFc = (n10c - n10s)/shock


    # thingtoplot = n10c
    # fig = figure(figsize=(8,6))
    # #ax = fig[:gca](projection="3d")
    # ax = Axes3D(fig)
    # ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
    # xgrid = repmat(cgrid,1,dur);
    # ygrid = repmat((1:dur)',nbins,1);
    # ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
    #                   edgecolors="k",linewidth=0.25)
    # xlabel("w")
    # ylabel("t")
    # zlabel("c_t(w)")
    # title("response of consumption distribution to Z")
    # savefig("cdist.eps")


    thingtoplot = qIRFc
    fig = figure(figsize=(8,6))
    #ax = fig[:gca](projection="3d")
    ax = Axes3D(fig)
    ax[:set_zlim](minimum(thingtoplot), maximum(thingtoplot));
    xgrid = repmat(cgrid,1,dur);
    ygrid = repmat((1:dur)',nbins,1);
    ax[:plot_surface](xgrid, ygrid, thingtoplot, cmap=ColorMap("jet"),
                      edgecolors="k",linewidth=0.25)
    xlabel("w")
    ylabel("t")
    savefig("cdist.eps")
end
toc()