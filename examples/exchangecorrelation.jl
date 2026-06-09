module XCEnergy

#              +----------------------------------------+
#>-------------|             Exchange                   |-------------<
#              +----------------------------------------+

const A = 1.071295
const a = 2.385345
const kappa = 1.0/a

#Exchange
function exun(n::Float64)
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
vxun(n::Float64) = n > 1.0e-12 ? -A*atan(a*n*pi)/pi : 0.0

#TEB: added just for completeness
vxpol(n::Float64) = n > 1.0e-12 ? -A*atan(2*a*n*pi)/pi : 0.0

#Derivative of exchange with respect to 1) nup 2) ndn
vxup(nup::Float64, ndn::Float64) = vxpol(nup)
vxdn(nup::Float64, ndn::Float64) = vxpol(ndn)

#              +----------------------------------------+
#>-------------|             Correlation                |-------------<
#              +----------------------------------------+

function poly(x,a0,a,b,c,d,e,f)
    a0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))
end

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

function polypol(n::Float64)
    y = pi * a * n
    poly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A)
end

function dpolypol(n::Float64)
    y = pi * a * n
    dpoly(sqrt(y),Palpha,Pbeta,Pgamma,Pdelta,Peta,Psigma,Pnu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 
end


ecpol(n::Float64) = -A * a * n^2 / polypol(n)

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

polyun(n::Float64) = poly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A)

dpolyun(n::Float64) = dpoly(sqrt(pi*a*n),alpha,beta,gamma,delta,eta,sigma,nu*pi*kappa^2/A) * sqrt(pi * a/n) * 0.5 

ecun(n::Float64) = -A * a * n^2 / polyun(n)

vcun(n::Float64) = n > 1.01e-12 ? 2 * ecun(n) / n + A * a * n^2 / polyun(n)^2 * dpolyun(n) : 0.0

function ec(nup::Float64,ndn::Float64)
    n = nup + ndn
    zeta = (nup - ndn)/n
    ecun(n) + zeta^2 * (ecpol(n) - ecun(n))
end

#used in program!!
function vc(nup::Float64,ndn::Float64)
    n = nup + ndn
    A*a*(4*ndn*nup*dpolyun(n)/polyun(n)^2-4*ndn/polyun(n)
    +(nup-ndn)^2*dpolypol(n)/polypol(n)^2+2*(ndn-nup)/polypol(n))
end

export expol, vxpol, ecpol, vcpol, ex, vx, ecun, vcun, ec, vc, vxup, vxdn

end

using .XCEnergy

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
