
module oneDFT

import DSP
import LinearAlgebra

"""
  thomas_fermi_nonInt(mu,n,v,x)

Non-interacting Thomas-Fermi density `sqrt(6)/pi * sqrt(mu-v(x))`
"""
function thomas_fermi_nonInt(mu::Float64,n::Array{W,1},v::Array{W,1},x::Integer) where W <: Number
  return sqrt(6.)/pi * sqrt(mu - v[x])
end
export thomas_fermi_nonInt

"""
  thomas_fermi(mu,n,v,x[,interaction=exp_int])

Interacting Thomas-Fermi density `sqrt(6)/pi * sqrt(mu-v(x))`
"""
function thomas_fermi(mu::Float64,n::Array{W,1},v::Array{W,1},x::Integer;interaction::Function=exp_int) where W <: Number
  vH_v = [vH(n,x,interaction=interaction) - v[x] for x = 1:length(n)]
  return sqrt(6)/pi .* sqrt.(mu .- vH_v)
end

function thomas_fermi(mu::Float64,n::Array{W,1},v::Array{W,1},x::Integer,interaction::Array{W,1}) where W <: Number
  vH_v = vH(n,interaction)
  for i = 1:length(v)
    vH_v[i] += vH_v[i] - v[i]
  end
  return sqrt(6)/pi .* sqrt.(mu .- vH_v)
end
export thomas_fermi

"""
  exp_int(x[,Aval=1,kappa=1])

Exponential interaction of the form `A*exp(-kappa*abs(x))`
"""
function exp_int(x::Number;Aval::Number=1.071295,kappa::Number=1/2.38345)
  return Aval * exp(-kappa * abs(x))
end
export exp_int

"""
  sC_int(x[,a=1])

Soft-Coulomb interaction of the form `1/sqrt(x^2+a^2)` at position `x` (Real number) with softening constant `a`

See also [`alt_sC_int`](@ref)
"""
function sC_int(x::Number;a::Number=1)
  return 1/sqrt(a^2+x^2)
end
export sC_int

"""
  alt_sC_int(x[,a=1])

Soft-Coulomb interaction of the form `1/(1+x^2)` at position `x` (Real number) with softening constant `a`

See also [`sC_int`](@ref)
"""
function alt_sC_int(x::Number;a::Number=1)
  return 1/(a^2+x^2)
end
export alt_sC_int

"""
  vH(density,x[,interaction=exp_int])

Computes the Hartree potential (note: does not multiply extra grid spacing factor)

Warning: where possible, use a convolution on a fixed vector input for the interaction vector instead of this function
"""
function vH(density::Array{W,1},x::Number;interaction::Function=exp_int) where W <: Number
  Ns = length(density)
  return sum(w->interaction(x-w) * density[w],1:Ns)
end

"""
  vH(density,interaction)

Computes the Hartree potential from a fixed vector for the interaction (note: does not multiply extra grid spacing factor)
"""
function vH(density::Array{W,1},interaction::Array{W,1}) where W <: Number
  Ns = length(density)
  convdens = DSP.conv(interaction,density)
  return convdens[Ns:2*Ns-1]
end
export vH


"""
  find_mu(n,N[,mu=1,dmu=0.1,N_tol=1E-10])

Obtains the chemical potential in a Thomas-Fermi iteration for a given problem
"""
function find_mu(n::Array{W,1},N::Integer;mu::Number=1.0,dmu::Number=0.1,N_tol::Number=1E-10) where W <: Number
  finding_mu = true
  while finding_mu
    temp = thomas_fermi(mu,n)
    Ncalc = sum(i->pi^2/6 * n[i] + temp[i],1:length(temp)) #no grid spacing here
    diffN = abs(Ncalc-N)
#    diffN = abs(dN)
    if diffN < N_tol
      finding_mu = false
    elseif Ncalc > N
      mu -= diffN * dmu
    else #if Ncalc <= N
      mu += diffN * dmu
    end
  end
  return mu
end
export find_mu


"""
  solveKS(v,[,Nex=1])

Solves Kohn-Sham system for a given input potential `v` for a number of excitations `Nex`. Will only compute the lowest-lying (`Nex`) eigenvectors to return energies and eigenvectors.
"""
function solveKS(H::AbstractArray;Nex::Integer=1)
  Ns = size(H,1)

  En = LinearAlgebra.eigvals(H)
  vecs = Array{Array{eltype(H),1},1}(undef,Nex)

  Threads.@threads for i = 1:Nex
    @inbounds vecs[i] = reshape(LinearAlgebra.eigvecs(H,[En[i]]),Ns)
  end
  return En[1:Nex],vecs
end
export solveKS

"""
  makedensity(vecs,Nup,Ndn)

Generates the density from the eigenvectors (`vecs`) of the given Kohn-Sham system for both `Nup` up electrons and `Ndn` down electrons. (assumes spin degeneracy in the orbitals)
"""
function makedensity(vecs::Array{Array{W,1},1},Nup::Integer,Ndn::Integer) where W <: Number
  density = zeros(W,length(vecs[1]))
  for i = 1:Nup
    @simd for x = 1:length(vecs[i])
      @inbounds density[x] += abs2(vecs[i][x])
    end
  end
  for i = 1:Ndn
    @simd for x = 1:length(vecs[i])
      @inbounds density[x] += abs2(vecs[i][x])
    end
  end
  return density
end

"""
  makedensity(vecs,Ne)

Generates the density from the eigenvectors (`vecs`) of the given Kohn-Sham system for both `Nup` up electrons and `Ndn` down electrons. (assumes spin degeneracy in the orbitals)
"""
function makedensity(vecs::Array{Array{W,1},1},Ne::Integer) where W <: Number
  density = zeros(W,length(vecs[1]))
  for i = 1:Ne
    @simd for x = 1:length(vecs[i])
      @inbounds density[x] += abs2(vecs[i][x])
    end
  end
  return density
end
export makedensity

"""
  nonInt(density,v)

Returns simply the potential. General function form is used for inputing density and potential to make a Kohn-Sham potential
"""
function nonInt(density::Array{W,1}) where W <: Number
  return zeros(W,length(density))
end
export nonInt

"""
  Power(a,b)

Interface function for output of Mathematica file for derivative of a given function.  Computes `a^b`
"""
function Power(a,b)
  return a^b
end
export Power

"""
  exp_coeffs(polarized)

Returns exponential coefficients for the exponential interaction from T.E. Baker, et. al. Phys. Rev. B 91, 235141 (2015)
"""
function exp_coeffs(polarized::Bool)
  if polarized
    alpha = 180.891
    beta = -541.124
    gamma = 651.615
    delta = -356.504
    eta = 88.0733
    sigma = -4.32708
    nu = 8
  else
    alpha = 2
    beta = -1.00077
    gamma = 6.26099
    delta = -11.9041
    eta = 9.62614
    sigma = -1.48334
    nu = 1
  end
  return alpha,beta,gamma,delta,eta,sigma,nu
end
export exp_coeffs

"""
  expLDA(density,v[,kappa=0.5,Aval=1,polarized=false])

Generates the LDA potential of the exponential interaction from T.E. Baker, et. al. Phys. Rev. B 91, 235141 (2015)
"""
function expLDA(density::Array{W,1},v::Array{W,1};kappa::Float64=1/2.385345,Aval::Float64=1.071295,polarized::Bool=false) where W <: Number
  alpha,beta,gamma,delta,eta,sigma,nu = exp_coeffs(polarized)

  Ns = length(density)
  vexpLDA = Array{W,1}(undef,Ns)
  @simd for i = 1:Ns
    #y = pi/(2*kappa*rs)
    #rs = 1/(2*density[i]) from BSWB15
    @inbounds vexpLDA[i] = (Aval*Power(density[i],2)*((gamma*pi)/kappa + 
                            (2*eta*Power(pi,2)*density[i])/Power(kappa,2) + 
                            (3*nu*Power(pi,4)*Power(density[i],2))/(Aval*kappa) + 
                            (beta*pi)/(2. *kappa*Sqrt((pi*density[i])/kappa)) + 
                            (3*delta*pi*Sqrt((pi*density[i])/kappa))/(2. *kappa) + 
                            (5*pi*sigma*Power((pi*density[i])/kappa,1.5))/(2. *kappa)))/
                        (kappa*Power(alpha + (gamma*pi*density[i])/kappa + 
                            (eta*Power(pi,2)*Power(density[i],2))/Power(kappa,2) + 
                            (nu*Power(pi,4)*Power(density[i],3))/(Aval*kappa) + 
                            beta*Sqrt((pi*density[i])/kappa) + delta*Power((pi*density[i])/kappa,1.5) + 
                            sigma*Power((pi*density[i])/kappa,2.5),2)) - 
                        (2*Aval*density[i])/
                        (kappa*(alpha + (gamma*pi*density[i])/kappa + 
                            (eta*Power(pi,2)*Power(density[i],2))/Power(kappa,2) + 
                            (nu*Power(pi,4)*Power(density[i],3))/(Aval*kappa) + 
                            beta*Sqrt((pi*density[i])/kappa) + delta*Power((pi*density[i])/kappa,1.5) + 
                            sigma*Power((pi*density[i])/kappa,2.5)))
  end
  return vexpLDA
end
export expLDA

function expLDA_energy(density::Array{W,1},vext::Array{W,1};polarized::Bool=false,kappa::Float64=0.5,Aval::Float64=1.) where W <: Number
  Ns = length(vext)
  energy = 0.
  
  coeff_Ts = polarized ? pi^2/6 : pi^2/24
  coeff_U = Aval/kappa
  coeff_Ex = (polarized ? 0.5 : 1) * Aval * kappa/(2*pi^2)
  alpha,beta,gamma,delta,eta,sigma,nu = exp_coeffs(polarized)
  coeff_Ec = -Aval*kappa/pi^2
  coeff_RPA = pi*kappa^2/Aval
  polfactor = polarized ? 2 : 1
  @simd for i = 1:Ns
  #  Kinetic energy
    @inbounds energy += coeff_Ts * density[i]^3
  #  Hartree energy
    @inbounds energy += coeff_U * density[i]^2
  #  Exchange energy
    @inbounds rs = 1/(2*density[i])
    y = pi/(2*kappa*rs)
    @inbounds rs_two = 1/(2*(polfactor*density[i]))
    ytwo = pi/(2*kappa*rs)
    energy += coeff_Ex * (log(1+ytwo^2) - 2 * ytwo * atan(ytwo))
  #  Correlation energy
    energy += coeff_Ec * y^2/(alpha + beta*sqrt(y) + gamma*y + delta *y^(3/2) + eta*y^2 + sigma*y^(5/2) + nu * coeff_RPA * y^3)
  #  Potential energy
    @inbounds energy += density[i] * vext[i]
  end
  return energy
end
export expLDA_energy

function mixKS!(density::Array{W,1},v::Array{W,1},Nup::Int64,Ndn::Int64,
                lambda::Float64;Delta::Float64=1.,dlambda::Float64=0.01,
                niter::Int64=100,tol::Float64=1E-6,makepotential::Function=nonInt,
                energy_fct::Function=expLDA_energy) where W <: Number
  println("starting density mixing for convergence (lambda = ",lambda,")")
  t1 = time()
  numorbs = max(Nup,Ndn)
  counter = 1
  error = 10000.
  energy = 0.


  Ns = length(v)
  onsite = ones(Float64,Ns)
  offsite = -0.5*ones(Float64,Ns-1)
  H0 = LinearAlgebra.SymTridiagonal(onsite,offsite)/Delta^2

  while error > tol && (counter <= niter || niter == 0)
    vKS = makepotential(density)
    @simd for w = 1:length(vKS)
      @inbounds vKS[w] += v[w]
    end
    H = H0 + LinearAlgebra.Diagonal(vKS)
    vals,vecs = solveKS(H,Nex=numorbs)
    output_density = makedensity(vecs,Nup,Ndn)
    error = 0.
    @simd for i = 1:length(density)
      @inbounds error += density[i] - output_density[i]
      @inbounds density[i] *= (1-lambda)
      @inbounds density[i] += lambda * output_density[i]
    end
    KSenergy = 0.
    @simd for p = 1:Nup
      @inbounds KSenergy += vals[p]
    end
    @simd for p = 1:Ndn
      @inbounds KSenergy += vals[p]
    end
    t2 = time()
    println("iteration ",counter," (",t2-t1,"s): KS energy = ",KSenergy," and density error = ",error)
    counter += 1
  end
  energy = energy_fct(density,v) * Delta
  return density,energy
end
export mixKS!

"""
  computeU(dens,interactionvec)

Computes the Hartree energy for a given density (`dens`) and an interaction vector (`interactionvec`) of the form

interactionvec = [exp(-abs(deltaX)) for deltaX = -(Ns-1):Ns-1]
"""
function computeU(dens::Array{W,1},interactionvec::Array{W,1}) where W <: Float64
  Ns = length(dens)
  convdens = vH(dens,interactionvec)
  return (dens' * convdens)/2
end
export computeU


function invertdensity(density::Array{W,1}) where W <: Number
  rho_scr = rand(length(density))
  rho_scr /= sum(rho_scr)
  rho_scr *= sum(density)
  return invertdensity(rho_scr,density)
end

function invertdensity(rho_scr::Array{W,1},density::Array{W,1},Nup::Integer,Ndn::Integer,epsilon::W=0.01;makepotential::Function=nonInt,dU::Number=1E-3,interaction::Function=nonInt) where W <: Number
  currU = computeU(density,interactionvec)
  prevU = zeros(length(currU))
  Ns = length(density)
  while sum(w->abs(currU[w]-prevU[w]),1:length(currU)) > dUtol
    vKS = makepotential(rho_scr,interaction=interaction)
    
    energies,vecs = solveKS(vKS,Nex=max(Nup,Ndn))
    rho_vKS = makedensity(vecs,Nup,Ndn)
    @simd for w = 1:Ns
      @inbounds rho_KS[w] -= density[w]
      @inbounds prevU[w] = currU[w]
    end
    currU = computeU(rho_KS,interactionvec)
    firstint = vH(rho_KS,interactionvec)
    dU = epsilon * (rho_scr' * firstint)
    @simd for w = 1:Ns
      @inbounds rho_KS[w] *= epsilon
      @inbounds rho_scr[w] -= rho_KS[w]
    end
    for w = 1:Ns
      vKS[w] += rho_KS[w]
    end
  end
  return invertdensity(rho_scr,density)
end
export invertdensity

end
using .oneDFT



Ns = 10001

Nup = 1
Ndn = 0

Delta = 0.01

vext = [-exp_int((x-Ns/2+1)*Delta) for x = 1:Ns] #[-exp(-0.5*abs(x-Ns/2+1)) for x = 1:Ns+1]
density = rand(length(vext))

import LinearAlgebra
H = LinearAlgebra.SymTridiagonal(vext+ones(Ns)/Delta^2,-0.5*ones(Ns)/Delta^2)
D,U = LinearAlgebra.eigen(H)
println("Should be... ",D[1])




density,energy = mixKS!(density,vext,Nup,Ndn,1.,Delta=Delta)
