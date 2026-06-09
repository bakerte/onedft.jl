
const A = 1.071295
const a = 2.385345
const kappa = 1.0/a

#Exchange
"""
   exun(n)

unpolarized exchange energy for a given density value `n`
"""
function exun(n::Float64)
  y = pi * a * n
  A * kappa * (log(1+y^2) - 2 * y * atan(y)) / (2 * pi^2)
end

#TEB: added just in case
"""
   expol(n)

polarized exchange energy for a given density value `n`
"""
expol(n::Float64) = exun(2n)/2

#Exchange
"""
   ex(nup,ndn)

exchange energy for a given up-density value `nup` and down-density value `ndn`
"""
function ex(nup::Float64, ndn::Float64)
  n = nup + ndn
  zeta = (nup - ndn)/n
  res = 0.5 * (exun((1+zeta)*n) + exun((1-zeta)*n))
end

#find vx at a point
"""
   vxun(n)

contribution to the Kohn-Sham potential (unpolarized) for density value `n`
"""
vxun(n::Float64) = n > 1.0e-12 ? -A*atan(a*n*pi)/pi : 0.0

#TEB: added just for completeness
"""
   vxpol(n)

contribution to the Kohn-Sham potential (polarized) for density value `n`
"""
vxpol(n::Float64) = n > 1.0e-12 ? -A*atan(2*a*n*pi)/pi : 0.0

#Derivative of exchange with respect to 1) nup 2) ndn
"""
    vxup(nup,ndn)

Contribution of the Kohn-Sham potential from the exchange (up) of the up-density `nup` and down-density `ndn`
"""
vxup(nup::Float64, ndn::Float64) = vxpol(nup)

"""
    vxdn(nup,ndn)

Contribution of the Kohn-Sham potential from the exchange (down) of the up-density `nup` and down-density `ndn`
"""
vxdn(nup::Float64, ndn::Float64) = vxpol(ndn)

#              +----------------------------------------+
#>-------------|             Correlation                |-------------<
#              +----------------------------------------+

"""
    poly(x,a0,a,b,c,d,e,f)

A polynomial in `x` of order 6 with coefficients `a0`,`a`,`b`,`c`,`d`,`e`,`f`
"""
function poly(x,a0,a,b,c,d,e,f)
  a0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))
end

"""
    dpoly(x,a0,a,b,c,d,e,f)

Derivative of a polynomial in `x` of order 6 with coefficients `a0`,`a`,`b`,`c`,`d`,`e`,`f`
"""
function dpoly(x,a0,a,b,c,d,e,f)
  a + x * (2b + x * (3c + x * (4d + x * (5e + x * f*6))))
end

#polarized
const Palpha = 180.891
const Pbeta = -541.124
const Pgamma = 651.615
const Pdelta = -356.504
const Peta = 88.0733
const Psigma = -4.32708
const Pnu = 8

"""
    polypol(n)

Evaluation of the polynomial function `poly` for polarized coefficients at density value `n`

See also: [`poly`](@ref)
"""
function polypol(n::Float64)
  y = pi * a * n
  poly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A)
end

"""
    dpolypol(n)

Evaluation of the derivative of the polynomial function `dpoly` for polarized coefficients at density value `n`

See also: [`dpoly`](@ref)
"""
function dpolypol(n::Float64)
  y = pi * a * n
  dpoly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 
end

"""
    ecpol(n)

exchange correlation density at a density point `n` for the polarized functional
"""
ecpol(n::Float64) = -A * a * n^2 / polypol(n)

"""
    ecpol(n)

Contribution of the correlation to the Kohn-Sham potential at a density point `n` for the polarized functional
"""
function vcpol(n::Float64)
  n > 1.01e-12 ? 2 * ecpol(n) / n + A * a * n^2 / polypol(n)^2 * dpolypol(n) : 0.0
end

#unpolarized correlation polynomial parameters
const alpha = 2
const beta = -1.00077
const gamma = 6.26099
const delta = -11.9041
const eta = 9.62614
const sigma = -1.48334
const nu = 1

"""
    polyun(n)

Evaluation of the polynomial function `poly` for unpolarized coefficients at density value `n`

See also: [`poly`](@ref)
"""
polyun(n::Float64) = poly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A)

"""
    dpolyun(n)

Evaluation of the derivative of the polynomial function `dpoly` for unpolarized coefficients at density value `n`

See also: [`dpoly`](@ref)
"""
dpolyun(n::Float64) = dpoly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 

"""
    ecun(n)

unpolarized corrleation energy density for a given density `n`
"""
ecun(n::Float64) = -A * a * n^2 / polyun(n)

"""
    vcun(n)

functional derivative of the unpolarized corrleation energy density for a given density `n`
"""
vcun(n::Float64) = n > 1.01e-12 ? 2 * ecun(n) / n + A * a * n^2 / polyun(n)^2 * dpolyun(n) : 0.0

"""
    ec(nup,ndn)

Evaluation of the correlation energy density for an up-density point `nup` and down-density `ndn`
"""
function ec(nup::Float64,ndn::Float64)
  n = nup + ndn
  zeta = (nup - ndn)/n
  ecun(n) + zeta^2 * (ecpol(n) - ecun(n))
end

#used in program!!
"""
    vc(nup,ndn)

Evaluation of the correlation part of the Kohn-Sham potential for an up-density point `nup` and down-density `ndn`
"""
function vc(nup::Float64,ndn::Float64)
  n = nup + ndn
  A*a*(4*ndn*nup*dpolyun(n)/polyun(n)^2-4*ndn/polyun(n)
  +(nup-ndn)^2*dpolypol(n)/polypol(n)^2+2*(ndn-nup)/polypol(n))
end