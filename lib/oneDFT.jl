
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





function loadstorevecs!(storevecs::NTuple{2,Array{W,2}},vecs::Array{W,2}) where W <: Number
  for a = 1:2
    for w = 1:size(vecs,1)
      @simd for i = 1:size(vecs,2)
        @inbounds storevecs[a][w,i] = vecs[w,i]
      end
    end
  end
  nothing
end


"""
  solveKS(v,[,Nex=1])

Solves Kohn-Sham system for a given input potential `v` for a number of excitations `Nex`. Will only compute the lowest-lying (`Nex`) eigenvectors to return energies and eigenvectors.
"""
function solveKS(H::AbstractArray;Nex::Integer=1,storevecs::NTuple{2,Array{W,2}}=(zeros(size(H,1),Nex),zeros(size(H,1),Nex))) where W <: Number
  
  #Lanczos part

#  if isapprox(LinearAlgebra.norm(storevecs[1]),0)
#    println("full solution")
    energies = LinearAlgebra.eigvals(H)
    En = energies[1:Nex]
    vecs = LinearAlgebra.eigvecs(H,En)
    #=
  else
    println("Lanczos solution")
    nLanczos = 2

    alpha = Array{W,1}(undef,nLanczos)
    beta = Array{W,1}(undef,nLanczos-1)
    savepsi = Array{Array{W,1},1}(undef,nLanczos)
  
    savepsi[1] = startvec
    psi = savepsi[1]
    alpha[1] = psi' * H * psi
  
    savepsi[2] = (H - alpha[1]*LinearAlgebra.I) * psi
    beta[1] = LinearAlgebra.norm(savepsi[2])
    savepsi[2] /= beta[1]
    for k = 2:nLanczos-1
      psi = savepsi[k]
      alpha[k] = psi' * H * psi
      savepsi[k+1] = (H - alpha[k]*LinearAlgebra.I) * savepsi[k] - beta[k-1]*savepsi[k-1]
      beta[k] = LinearAlgebra.norm(savepsi[k+1])
      savepsi[k+1] /= beta[k]
    end
    psi = savepsi[end]
    alpha[end] = psi' * H * psi
  
  
    M = LinearAlgebra.SymTridiagonal(alpha,beta)
    D,U = LinearAlgebra.eigen(M)
  
    Ns = length(savepsi[1])  
    groundstate = Array{W,1}(undef,Ns)
    #sum(k->savepsi[k]*conj(U[k,1]),1:size(U,1)) #
  
    for x = 1:Ns
      groundstate[x] = 0
      for k = 1:size(U,1)
        groundstate[x] += conj(U[k,1])*savepsi[k][x]
      end
    end
  
    vecs = reshape(groundstate,Ns,1)
    En = D[1:1]
    if beta[end] > 1E-6 #!isapprox(energies[1],D[1])
      println("")
      energies = LinearAlgebra.eigvals(H)
      En = energies[1:Nex]
      vecs = LinearAlgebra.eigvecs(H,En)
    end



  end=#
  loadstorevecs!(storevecs,vecs)
  return En,vecs,En,vecs
end
export solveKS

"""
  makedensity(vecs,Ne)

Generates the density from the eigenvectors (`vecs`) of the given Kohn-Sham system for either up or down electrons.
"""
function makedensity(vecs::Array{W,2}) where W <: Number
  density = zeros(W,size(vecs,1))
  for x = 1:size(vecs,1)
    @simd for i = 1:size(vecs,2)
      @inbounds density[x] += abs2(vecs[x,i])
    end
  end
  return density
end

function makedensity(vecs::Array{W,2},Ne::Integer) where W <: Number
  density = zeros(W,size(vecs,1))
  for x = 1:size(vecs,1)
    @simd for i = 1:Ne
      @inbounds density[x] += abs2(vecs[x,i])
    end
  end
  return density
end

"""
  makedensity(vecs,Nup,Ndn)

Generates the density from the eigenvectors (`vecs`) of the given Kohn-Sham system for both `Nup` up electrons and `Ndn` down electrons. (assumes spin degeneracy in the orbitals)
"""
function makedensity!(newdens::Array{W,1},vecs::Array{W,2},Nup::Integer,Ndn::Integer) where W <: Number
  for x = 1:size(vecs[i],1)
    @simd for i = 1:min(Nup,Ndn)
      @inbounds newdens[x] += 2*abs2(vecs[x,i])
    end
  end
  interval = max(min(Nup,Ndn),1):abs(Nup-Ndn)
  for x = 1:length(vecs[i])
    @simd for i = interval
      @inbounds newdens[x] += abs2(vecs[x,i])
    end
  end
  return newdens
end
export makedensity!

function makedensity(vecs::Array{W,2},Nup::Integer,Ndn::Integer) where W <: Number
  newdens = zeros(W,size(vecs,1))
  return makedensity!(newdens,vecs,Nup,Ndn)
end

function makedensity(upvecs::Array{W,2},Nup::Integer,dnvecs::Array{W,2},Ndn::Integer) where W <: Number
  updens = makedensity(upvecs,Nup)
  dndens = makedensity(dnvecs,Ndn)
  return updens,dndens
end

function makedensity(upvecs::Array{W,2},dnvecs::Array{W,2}) where W <: Number
  updens = makedensity(upvecs)
  dndens = makedensity(dnvecs)
  return updens,dndens
end
export makedensity

"""
  nonInt(density,v)

Returns simply the potential. General function form is used for inputing density and potential to make a Kohn-Sham potential
"""
function nonInt(Delta::Number,density::Array{W,1},extra...) where W <: Number
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
  if !polarized
    alpha = 2
    beta = -1.00077
    gamma = 6.26099
    delta = -11.9041
    eta = 9.62614
    sigma = -1.48334
    nu = 1
  else
    alpha = 180.891
    beta = -541.124
    gamma = 651.615
    delta = -356.504
    eta = 88.0733
    sigma = -4.32708
    nu = 8
  end
  return alpha,beta,gamma,delta,eta,sigma,nu
end
export exp_coeffs

#=
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
=#

function expLDA_energy(Delta::Number,orbup::Array{W,2},orbdn::Array{W,2},H0::AbstractArray,
                       density::Array{W,1},vext::Array{W,1},interactionvec::Array{W,1};polarized::Bool=false,
                       Aval::Number=1.071295,kappa::Number=1/2.38345) where W <: Number
  
  Ns = length(density)
  A = Aval

  #=
  Nup = size(orbup,2)
  Ts_up = orbup'*H0*orbup
  Ndn = size(orbdn,2)
  Ts_dn = orbdn'*H0*orbdn
  Ts = sum(w->Ts_up[w,w],1:Nup) + sum(w->Ts_dn[w,w],1:Ndn)
  =#
  Ts = 0.
  for w = 1:size(orbup,2)
    currorb = orbup[:,w]
    Ts += currorb' * H0 * currorb
  end
  for w = 1:size(orbdn,2)
    currorb = orbdn[:,w]
    Ts += currorb' * H0 * currorb
  end
  Ts /= -2
  Ts *= Delta

  U = computeU(density,interactionvec) * Delta^2

  Ex = 0
  @simd for i = 1:Ns
    @inbounds rs = 1/(2*density[i])
    y = pi/(2*kappa*rs)
    Ex += A*kappa*(log(1+y^2)-2*y*atan(y))/(2*pi^2)
  end
  Ex *= Delta

  alpha,beta,gamma,delta,eta,sigma,nu = exp_coeffs(polarized)

  Ec = 0
  @simd for i = 1:Ns
    @inbounds rs = 1/(2*density[i])
    y = pi/(2*kappa*rs)
    Ec += -A*kappa * y^2/pi^2 / (alpha + beta*sqrt(y) + gamma*y + delta*y^(3/2) + eta*y^2 + sigma*y^(5/2) + nu * pi * kappa^2/A * y^3)
  end
  Ec *= Delta

  V = (density' * vext) * Delta
  return Ts + U + Ex + Ec + V
end
export expLDA_energy

function ArcTan(x)
  return atan(x)
end

const Pi = pi;

function Sqrt(x)
  return sqrt(x)
end

function expLDA_vHxc(Delta::Number,density::Array{W,1},interactionvec::Array{W,1};
                     polarized::Bool=false,Aval::Number=1.071295,
                     kappa::Number=1/2.38345) where W <: Number
  
  Ns = length(density)
  vHxc = vH(density,interactionvec)*Delta
  A = Aval
  @simd for i = 1:Ns
    @inbounds n = density[i]
    @inbounds vHxc[i] += -((A*ArcTan((n*Pi)/kappa))/Pi)
  end
  alpha,beta,gamma,delta,eta,sigma,nu = exp_coeffs(polarized)
  @simd for i = 1:Ns
    @inbounds n = density[i]
    @inbounds vHxc[i] += (3.1415926535897936*Power(A,2)*kappa*n*
    (-0.6366197723675813*A*alpha*Power(kappa,2) + 
      31.006276680299848*kappa*Power(n,3)*nu + 
      A*n*(-0.9999999999999994*gamma*kappa - 
         (0.8462843753216344*beta*kappa)/Power(n/kappa,0.5) + 
         2.784163998415857*kappa*Power(n/kappa,1.5)*sigma - 
         0.8862269254527569*kappa*Power(n/kappa,0.5)*delta + 2.1914924100062367e-15*n*eta)))/Power(1. *A*alpha*Power(kappa,2) + 
         97.40909103400243*kappa*Power(n,3)*nu + A*n*(3.141592653589793*gamma*kappa + 
       (1.772453850905516*beta*kappa)/Power(n/kappa,0.5) + 
       17.49341832762486*kappa*Power(n/kappa,1.5)*sigma + 
       5.568327996831707*kappa*Power(n/kappa,0.5)*delta + 9.869604401089358*n*eta),2)
  end
  return vHxc
end
export expLDA_vHxc

function TF_expLDA_energy(density::Array{W,1},vext::Array{W,1};polarized::Bool=false,kappa::Float64=0.5,Aval::Float64=1.) where W <: Number
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
export TF_expLDA_energy

function freeHamiltonian(Delta::Number,Ns::Integer)
  onsite = ones(Float64,Ns)
  offsite = -0.5*ones(Float64,Ns-1)
  H0 = LinearAlgebra.SymTridiagonal(onsite,offsite)/Delta^2
  return H0
end
export freeHamiltonian

function updatedensity!(lambda::W,density::Array{W,1},newdens::Array{W,1}) where W <: Number
  DeltaDens = newdens - density
  Ns = length(density)
  error = 0.
  @simd for i = 1:Ns
    @inbounds density[i] += DeltaDens[i] * lambda
    @inbounds error += abs(DeltaDens[i])
  end
  return error/Ns
end
export updatedensity!




function totKSenergy(upvals::Array{W,1},Nup::Integer) where W <: Number
  KSenergy = 0.
  @simd for p = 1:Nup
    @inbounds KSenergy += upvals[p]
  end
  return KSenergy
end


function densnorm!(density::Array{W,1},Ne::Integer,Delta::Number) where W <: Number
  densnorm = Ne/(sum(density)*Delta)
  @simd for i = 1:length(density)
    @inbounds density[i] *= densnorm
  end
  nothing
end


using Plots




import Printf
function mixKS!(density::Array{W,1},v::Array{W,1},Nup::Int64,Ndn::Int64,
                lambda::Float64;Delta::Float64=1.,dlambda::Float64=0.01,
                niter::Int64=0,tol::Float64=1E-5,vHxc::Function=nonInt,
                energy_fct::Function=TF_expLDA_energy,Ns::Integer = length(v),
                storevecs::NTuple{2,Array{W,2}}=(zeros(length(v),Nup),zeros(length(v),Ndn)),
                interactionvec = [exp_int(-abs(Delta * deltaX)) for deltaX = -(Ns-1):Ns-1]) where W <: Number
  println("starting density mixing for convergence (lambda = ",lambda,")")
  numorbs = max(Nup,Ndn)
  counter = 1
  error = 10000.
  energy = 0.



  Ne = Nup+Ndn
  densnorm!(density,Ne,Delta)



  H0 = freeHamiltonian(Delta,Ns)

#  vKS_last = Array{Float64,1}(undef,Ns)
#  density_last = Array{Float64,1}(undef,Ns)

  
  energy = 0.
  while error > tol && (counter <= niter || niter == 0)


    t1 = time()
    vKS = vHxc(Delta,density,interactionvec)

    @simd for w = 1:Ns
      @inbounds vKS[w] += v[w]
    end
    H = H0 + LinearAlgebra.Diagonal(vKS)
    upvals,orbsup,dnvals,orbsdn = solveKS(H,Nex=numorbs,storevecs=storevecs)

    updensity,dndensity = makedensity(orbsup,Nup,orbsdn,Ndn)
    output_density = updensity + dndensity

    densnorm!(output_density,Ne,Delta)
    error = updatedensity!(lambda,density,output_density)


    if false #counter > 1
#      clf()
#      plot(vKS_last,"r--",label="vKS previous")
      plot(vKS,"b-",label="vKS")
      plot!(vKS+v,"g-.",label="vKS+vext")
      plot!(v,"k--",label="vext")
#      plot(output_density/Delta,"m-",label="density")
#      plot(density_last/Delta,"m--",label="previous density")
#      newdensity = (1-lambda)*density_last + lambda*output_density
#      plot(newdensity/Delta,"m-.",label="density should be")
      plot!(density#=/Delta=#,"c-",label="current mixed density")
      legend(fontsize=10,borderaxespad=0.5,loc="lower left",edgecolor="black")
#=
      @simd for i = 1:Ns
        @inbounds vKS_last[i] = vKS[i]
        @inbounds density_last[i] = density[i]
      end
      =#
    end

    KSenergy = totKSenergy(upvals,Nup) + totKSenergy(dnvals,Ndn)
    energy = energy_fct(Delta,orbsup,orbsdn,H0,density,v,interactionvec)

    t2 = time()
    println("iteration ",counter," (",Printf.@sprintf("%.2f",t2-t1),"s): KS energy = ",KSenergy,", true energy ",energy,", and density error = ",error)
    println()
    counter += 1
  end
  return energy,density
end
export mixKS!

"""
  computeU(dens,interactionvec)

Computes the Hartree energy for a given density (`dens`) and an interaction vector (`interactionvec`) of the form

interactionvec = [exp(-abs(deltaX)) for deltaX = -(Ns-1):Ns-1]
"""
function computeU(dens::Array{W,1},interactionvec::Array{W,1}) where W <: Float64
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
    dU = epsilon * LinearAlgebra.dot(rho_scr,firstint)
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