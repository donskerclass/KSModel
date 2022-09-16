function sspolicy(params, K, L, wgrid, sgrid, wweights, sweights, Win, g)
    # outputs: [c,ap,Wout,tr]

    @unpack nw, ns, β, γ, α, δ, smoother, sigs, zhi, zlo = params

    c = zeros(nw)      # consumption
    ap = zeros(nw)      # savings
    damp = 0.8          # how much we update W guess
    count = 1
    dist = 1.0          # distance between Win and Wout on decision rules
    tol = 1e-5          # acceptable error 
    Wout = copy(Win)    # initialize Wout
    R = α * (K^(α - 1.0)) * (L^(1.0 - α)) + 1.0 - δ
    wage = (1.0 - α) * (K^α) * (L^(-α))
    tr = zeros(nw, nw)   # transition matrix mapping todays dist of w to w'
    q = zeros(nw, nw * ns) # auxilary variable to compute LHS of euler
    while dist > tol && count < 5000
        # compute c(w) given guess for Win = β*R*E[u'(c_{t+1})]
        c = min.(Win .^ (-1.0 / γ), wgrid)
        ap = wgrid - c  # compute ap(w) given guess for Win
        for iw = 1:nw
            for iwp = 1:nw
                for iss = 1:ns
                    q[iw, nw*(iss-1)+iwp] = mollifier((wgrid[iwp] - R * ap[iw]) / wage - sgrid[iss], zhi, zlo, smoother)
                end
            end
        end
        Wout = β * R * q * kron(sweights .* (g / wage), wweights .* (c .^ (-γ)))
        dist = maximum(abs.(Wout - Win))
        Win = damp * Wout + (1.0 - damp) * Win
        count += 1
    end
    if count == 5000
        @warn "Euler iteration did not converge"
    end

    for iw = 1:nw
        for iwp = 1:nw
            sumns = 0.0
            for isp = 1:ns
                sumns += mollifier((wgrid[iwp] - R * ap[iw]) / wage - sgrid[isp], zhi, zlo, smoother) * (g[isp] / wage) * sweights[isp]
            end
            tr[iwp, iw] = sumns
        end
    end
    return (c, ap, Wout, tr)
end

function findss(params,
    K,
    L,
    wgrid,
    sgrid,
    wweights,
    sweights,
    Win, MW, MWinv,
    g)

    @unpack ns, nw, β, γ, α, δ, smoother, sigs, zhi, zlo = params

    tol = 1 - 5
    count = 1
    maxit = 100

    c = zeros(nw)
    ap = zeros(nw)
    KF = zeros(nw, nw)
    wdist = zeros(nw)
    diseqm = 1234.5
    newK = 0.0
    Kdamp = 0.1
    while abs(diseqm) > tol && count < maxit # clearing markets
        (c, ap, Win, KF) = sspolicy(params, K, L, wgrid, sgrid, wweights, sweights, Win, g)
        # sspolicy returns the consumption decision rule, a prime
        # decision rule, Win = updated β R u'(c_{t+1}), 
        # KF is the Kolmogorov foward operator
        # and is the map which moves you from cash on hand distribution
        # today to cash on had dist tomorrow

        LPMKF = MW * KF * MW'

        # find eigenvalue closest to 1
        (D, V) = eigen(LPMKF)
        if abs(D[end] - 1) > 2e-1 # that's the tolerance we are allowing
            @warn "your eigenvalue is too far from 1, something is wrong"
        end
        wdist = MWinv * real(V[:, end]) #Pick the eigen vector associated with the largest
        # eigenvalue and moving it back to values

        wdist = wdist / (wweights' * wdist) #Scale of eigenvectors not determinate: rescale to integrate to exactly 1
        newK = wweights' * (wdist .* ap)  #compute excess supply of savings, which is a fn of w
        diseqm = newK - K
        #println(newK)
        K += Kdamp * diseqm
        count += 1
    end
    return (Win, c, wdist, newK, KF)
end
