module XCEnergy

#              +----------------------------------------+
#>-------------|             Exchange                   |-------------<
#              +----------------------------------------+

function expcoeffs()::NTuple{3,Float64}
  A = 1.071295
  a = 2.385345
  kappa = 1.0/a
  return A,a,kappa
end

#Exchange
function exun(n::Float64)
  A,a,kappa = expcoeffs()
  y = pi * a * n
  A * kappa * (log(1+y^2) - 2 * y * atan(y)) / (2 * pi^2)
end

#TEB: added just in case
expol(n::Float64) = exun(2n)/2

#Exchange
function ex(nup::Float64, ndn::Float64)
  n = nup + ndn
  zeta = (nup - ndn)/n
  res = 0.5 * (exun((1+zeta)*n) + exun((1-zeta)*n))
end

#find vx at a point
function vxun(n::Float64)
  A,a,kappa = expcoeffs()
  -A*atan(a*n*pi)/pi
end

#TEB: added just for completeness
function vxpol(n::Float64)
  A,a,kappa = expcoeffs()
  -A*atan(2*a*n*pi)/pi
end

#Derivative of exchange with respect to 1) nup 2) ndn
#vxup(nup::Float64, ndn::Float64) = vxpol(nup)
#vxdn(nup::Float64, ndn::Float64) = vxpol(ndn)

#              +----------------------------------------+
#>-------------|             Correlation                |-------------<
#              +----------------------------------------+

function poly(x::W,a0::W,a::W,b::W,c::W,d::W,e::W,f::W) where W <: Number
    a0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))
end

function dpoly(x::W,a0::W,a::W,b::W,c::W,d::W,e::W,f::W) where W <: Number
    a + x * (2b + x * (3c + x * (4d + x * (5e + x * f*6))))
end

function polcoeffs()::NTuple{7,Float64}
  #polarized
  Palpha = 180.891
  Pbeta = -541.124
  Pgamma = 651.615
  Pdelta = -356.504
  Peta = 88.0733
  Psigma = -4.32708
  Pnu = 8.
  return Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu
end

function unpolcoeffs()::NTuple{7,Float64}
  #unpolarized correlation polynomial parameters
  alpha = 2.
  beta = -1.00077
  gamma = 6.26099
  delta = -11.9041
  eta = 9.62614
  sigma = -1.48334
  nu = 1.
  return alpha,beta,gamma,delta,eta,sigma,nu
end

function polypol(n::Float64)
  A,a,kappa = expcoeffs()
  Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu = polcoeffs()
  y = pi * a * n
  poly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A)
end

function dpolypol(n::Float64)
  A,a,kappa = expcoeffs()
  Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu = polcoeffs()
  y = pi * a * n
  dpoly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 
end


function ecpol(n::Float64)
  A,a,kappa = expcoeffs()
  -A * a * n^2 / polypol(n)
end

function vcpol(n::Float64)
  A,a,kappa = expcoeffs()
  2 * ecpol(n) / n + A * a * n^2 / polypol(n)^2 * dpolypol(n)
end

function polyun(n::Float64)
  A,a,kappa = expcoeffs()
  alpha,beta,gamma,delta,eta,sigma,nu = unpolcoeffs()
  poly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A)
end

function dpolyun(n::Float64)
  A,a,kappa = expcoeffs()
  alpha,beta,gamma,delta,eta,sigma,nu = unpolcoeffs()
  dpoly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 
end

function ecun(n::Float64)
  A,a,kappa = expcoeffs()
  -A * a * n^2 / polyun(n)
end

function vcun(n::Float64)
  A,a,kappa = expcoeffs()
  2 * ecun(n) / n + A * a * n^2 / polyun(n)^2 * dpolyun(n)
end

function ec(nup::Float64,ndn::Float64)
  n = nup + ndn
  zeta = (nup - ndn)/n
  ecun(n) + zeta^2 * (ecpol(n) - ecun(n))
end

#used in program!!
function vc(nup::Float64,ndn::Float64)
  A,a,kappa = expcoeffs()
  n = nup + ndn
  A*a*(4*ndn*nup*dpolyun(n)/polyun(n)^2-4*ndn/polyun(n)
  +(nup-ndn)^2*dpolypol(n)/polypol(n)^2+2*(ndn-nup)/polypol(n))
end

export expol, vxpol, ecpol, vcpol, ex, vx, ecun, vcun, ec, vc, vxup, vxdn

end

using Main.XCEnergy

#=
Usage::
    for i = 1:ndim
        vKSup[i] = v(i*Delta) + vH[i] + vxup(currdensup[i],currdensdn[i]) + vc(currdensup[i],currdensdn[i])
        vKSdn[i] = v(i*Delta) + vH[i] + vxdn(currdensup[i],currdensdn[i]) + vc(currdensdn[i],currdensup[i])
    end


    ehenergy = U(currdens)
    exenergy = sum(i->ex(currdensup[i],currdensdn[i])*Delta,1:ndim)	#exchange energy
    ecenergy = sum(i->ec(currdensup[i],currdensdn[i])*Delta,1:ndim)	#correlation
    venergy = sum(i->currdens[i]*v(i*Delta)*Delta,1:ndim)		#external potential energy
    energy=tsenergyup+tsenergydn+ehenergy+exenergy+ecenergy+venergy
    println("Energy: ",energy)
    println("Ts[n] = ",tsenergyup+tsenergydn,
	" | Ts[nup] = ",tsenergyup," | Ts[ndn] = ",tsenergydn," | U[n] = ",ehenergy)
    println("Ex[n] = ",exenergy," | Ec[n] = ",ecenergy," | V[n] = ",venergy)
=#
