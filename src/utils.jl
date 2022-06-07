# kind = 2
function chebpts(n, lo, hi)
    x = reverse(clenshawcurtisnodes(Float64, n))
    xr = ((hi - lo) / 2) * x .+ (hi + lo) / 2

    # could use clenshawcurtisweights for weights
    # but not sure about modified Chebyshev moments
    tvec = 1 .- [2.0:2.0:(n-1)...] .^ 2
    tk = zeros(size(tvec, 1) + 1, 1)
    tk[1, 1] = 1
    tk[2:end, 1] .= tvec
    c = 2 ./ tk'
    cvec = c[floor(Int, n / 2):-1:2]'
    ck = zeros(1, size(c, 2) + size(cvec, 2))
    # = and not .= here 
    ck[1, 1:size(c, 2)] = c
    ck[1, size(c, 2)+1:end] = cvec # Mirror for DCT via FFT
    intw = real(ifft(ck, 2))
    zz = intw[1] / 2
    w = zeros(1, n)
    w[1:size(intw, 2)] = intw
    w[1] = zz
    w[n] = zz

    xwts = ((hi - lo) / 2) * w

    return (cgrid=xr, cwts=xwts)
end

function grids(params)
    @unpack wlo, whi, nw, ns, slo, shi = params
    wscale = whi - wlo
    wgrid = collect(range(wlo, whi, length=nw))
    wweights = (wscale / nw) * ones(nw)

    scheb = chebpts(ns, slo, shi)
    sgrid = scheb[1]'
    sweights = dropdims(scheb[2]', dims=2)

    MW = sqrt(wscale / nw) * I(nw)
    MWinv = sqrt(nw / wscale) * I(nw)


    return wgrid, wweights, scheb, sgrid, sweights, MW, MWinv
end

function mollifier(z, zhi, zlo, smoother)
    if z < zhi && z > zlo
        temp = -1.0 + 2.0 * (z - zlo) / (zhi - zlo)
        temp = temp / smoother
        out = ((zhi - zlo) / 2.0) * exp(-1.0 / (1.0 - temp * temp)) / (In * smoother)
    else
        out = 0.0
    end
    return out
end

# differs by 2 at the very last decimal place
function trunc_lognpdf(x, UP, LO, mu, sig)
    # truncated log normal pdf
    logn = LogNormal(mu, sig)
    return (x - LO .>= -eps()) .* (x - UP .<= eps()) .* pdf.(logn, x) / (cdf(logn, UP) - cdf(logn, LO))
end

MPK(Z, K) = α * Z * (K^(α - 1.0)) * (L^(1.0 - α)) + 1.0 - δ   # Marginal product of capital + 1-δ
MPL(Z, K) = (1.0 - α) * Z * (K^α) * (L^(-α))  # Marginal product of labor
