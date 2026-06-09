#using DMRJtensor
#using TensorPACK
#using QuadGK

import DSP
function convolve(x::Vector{W},vsmall::Vector{W}) where W <: Number
  res = DSP.conv(x,vsmall)
  Ns = length(x)
  return res[1+Ns:2Ns] #W[Delta*res[i+Ns] for i=1:length(x)]
end

# The following lines implement the Kohn-Sham potential algorithm

# A function to calculate the ground state density ρv for a given potential v
import LinearAlgebra
function compute_orbitals(v,H0,Nup,Ndn)
  H = H0 + LinearAlgebra.Diagonal(v)
  lambdaup = LinearAlgebra.eigvals(H,1:max(Nup,Ndn))
  return LinearAlgebra.eigvecs(H,lambdaup)
end
export compute_orbitals

"""
    calculate_density(v,H0,Delta,Nup,Ndn)

Computes the density of a given external potential `v`, non-interacting Hamiltonian `H0`, grid spacing `Delta`, and number of up (down) fermions `Nup` (`Ndn`)
"""
function calculate_density(vecs,Nup,Ndn)

  Ns = size(vecs, 1)

  rhoKS = Array{Float64,1}(undef,Ns)
  for w = 1:Ns
    updens = Nup > 0 ? sum(k->vecs[w,k] .^ 2,1:Nup) : 0.
    dndens = Ndn > 0 ? sum(k->vecs[w,k] .^ 2,1:Ndn) : 0.

    rhoKS[w] = updens + dndens
  end
  return rhoKS
end

function calculate_density(v,H0,Delta,Nup,Ndn)
  vecs = compute_orbitals(v,H0,Nup,Ndn)
  calculate_density(vecs,Nup,Ndn)
end

#ThomasFermi(n::Array{Float64,1}) = 

vexp(x::Float64) = A*exp(-kappa*abs(x)/2)
"""
   findvKS(nexact)

Computes the exact Kohn-Sham potential (up to a constant) from an input exact interactin density (nexact)
"""
function findvKS(ndmrg,Delta,Ne_up,Ne_dn;Ns=length(ndmrg),vKS=-(ndmrg)/(Ne_up+Ne_dn)#=ThomasFermi=#,goal=1E-6,mix=0.3)#,iter=1000)
  ts = -1/(2*Delta^2)*ones(Ns)
  ts_onsite = 1/Delta^2*ones(Ns)
  H0 = LinearAlgebra.SymTridiagonal(ts_onsite,ts)

  vsmall = [vexp((k-Ns-1)*Delta) for k=1:2*Ns+1]

  nKS = Array{Float64,1}(undef,Ns)
  count = 0
  # Perform iterations until convergence
  diff_up = Array{Float64,1}(undef,Ns)
  while count == 0 || abs(sum(diff_up)/Ns) > goal #|| count < iter
    count += 1

    println(count," ",abs(sum(diff_up))/Ns)

    checkup = calculate_density(vKS,H0,Delta,Ne_up,Ne_dn)
    @inbounds @simd for w = 1:Ns
      nKS[w] = checkup[w]
    end

    diffdens_up = ndmrg-nKS
    integralup = convolve(diffdens_up,vsmall)
    @inbounds @simd for w = 1:Ns
      vKS[w] -= mix*integralup[w]
    end

    @inbounds @simd for w = 1:length(diff_up)
      diff_up[w] = abs(nKS[w] - ndmrg[w])
    end

  end
  return vKS,nKS
end
export findvKS