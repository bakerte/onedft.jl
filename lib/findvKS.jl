using DMRJtensor
using TensorPACK
#using QuadGK

import DSP
function convolve(x::Vector{W},vsmall::Vector{W}) where W <: Number
  res = DSP.conv(x,vsmall)
  Ns = length(x)
  return res[1+Ns:2Ns] #W[Delta*res[i+Ns] for i=1:length(x)]
end

# The following lines (6-66) calculate the original density matrix from interacting Hubbard model

Ns = 11
Ne = 2
Ne_up = cld(Ne,2)
Ne_dn = Ne-Ne_up
QS = 4
Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

psi = MPS(QS,Ns) 

upsites = [i+fld(Ns,2) for i = 1:Ne_up]
Cupdag = Cup'
applyOps!(psi,upsites,Cupdag,trail=F)

dnsites = [i for i = 1:Ne_dn]
Cdndag = Cdn'
applyOps!(psi,dnsites,Cdndag,trail=F)

Delta = 0.05
mu = 1.0/(Delta^2)
HubU = 0.0
t = -1.0/(2*Delta^2)

c = cld(Ns,2)
vext = [-exp(-Delta*abs(i-(1.05*c))/2)-exp(-Delta*abs(i-c)/2) for i = 1:Ns]

println("HERE")

function makeHubbard(Ns,t,HubU,mu,Cup,Cdn,F,Nup,Ndn,Ndens)
  mpo = 0
  for i = 1:Ns-1
    #spin up
    mpo += mpoterm(-t,Cup,i,Cup',i+1,F)
    mpo += mpoterm(t,Cup',i,Cup,i+1,F)
    #spin down
    mpo += mpoterm(-t,Cdn,i,Cdn',i+1,F)
    mpo += mpoterm(t,Cdn',i,Cdn,i+1,F)
  end
  for i = 1:Ns
    mpo += mpoterm(mu,Ndens,i)
    mpo += mpoterm(vext[i],Ndens,i)
  end

  return MPO(mpo) #+ expMPO(exp(-Delta/2),Ndens,Ndens,Ns)
end

mpo = makeHubbard(Ns,t,HubU,mu,Cup,Cdn,F,Nup,Ndn,Ndens)

@makeQNs "fermion" U1 U1
Qlabels = [fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]
qmpo,qpsi = MPO(Qlabels,mpo,psi) 

EnergyQN = dmrg(qpsi,qmpo,m=2,sweeps=300,goal=1E-2,cutoff=1E-9,method="twosite")

EnergyQN = dmrg(qpsi,qmpo,m=40,sweeps=300,goal=1E-8,cutoff=1E-9,method="twosite")

qCup,qCdn,qNup,qF,qNdens,qNdn = Qtens(Qlabels,Cup,Cdn,Nup,F,Ndens,Ndn)



if false
  rhoup = correlationmatrix(qpsi,qCup',qCup,trail=qF)
  rhodn = correlationmatrix(qpsi,qCdn',qCdn,trail=qF)
  ndmrg_up = [rhoup[w,w] for w = 1:Ns]
  ndmrg_dn = [rhodn[w,w] for w = 1:Ns]
elseif false
  ndmrg_up = correlation(qpsi,qNup)
  ndmrg_dn = correlation(qpsi,qNdn)
end

ndmrg = correlation(qpsi,qNdens)

# The following lines (75-end) implement the Kohn-Sham potential algorithm

# A function to calculate the ground state density ρv for a given potential v
import LinearAlgebra
function calculate_density(v,H0,Delta,Nup,Ndn)
  H = H0 + LinearAlgebra.Diagonal(v)
  lambdaup = LinearAlgebra.eigvals(H,1:max(Nup,Ndn))
  vecs = LinearAlgebra.eigvecs(H,lambdaup)


  rhoKS = Array{Float64,1}(undef,Ns)
  for w = 1:Ns
    updens = Nup > 0 ? sum(k->vecs[w,k] .^ 2,1:Nup) : 0.
    dndens = Ndn > 0 ? sum(k->vecs[w,k] .^ 2,1:Ndn) : 0.

    rhoKS[w] = updens + dndens
  end
  return rhoKS
end





c = cld(Ns,2)

vKS = copy(vext) #+ 0.01*rand(Ns)

#vKS = [-exp(Delta*abs(i-c)/2) for i = 1:Ns]

ThomasFermi = vKS


vexp(x::Float64) = exp(-abs(x)/2)
function findvKS(ndmrg;Ns=length(ndmrg),vKS=ThomasFermi,goal=1E-6,mix=0.3)
  ts = -1/(2*Delta^2)*ones(Ns)
  ts_onsite = 1/Delta^2*ones(Ns)
  H0 = LinearAlgebra.SymTridiagonal(ts_onsite,ts)

  vsmall = [vexp((k-Ns-1)*Delta) for k=1:2*Ns+1]

  nKS = Array{Float64,1}(undef,Ns)
  count = 0
  # Perform iterations until convergence
  diff_up = Array{Float64,1}(undef,Ns)
  while count == 0 || abs(sum(diff_up)/Ns) > goal
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

g = 50
gauss_guess = [-2*exp(-Delta/g*abs(i-c)^2/2) for i = 1:Ns]

vKS,nKS = findvKS(ndmrg,vKS=copy(gauss_guess))

using Plots

plot(nKS,label="noninteracting density",init=true)

plot!(vKS,label="KS Potential",init=true)

plot!(ndmrg,label="interact density",init=true)

plot!(vext,label="External Potential", init =true)

plot!(gauss_guess,label="initial guess",init=true)

En = expect(qpsi,qmpo)

sqrtdens = sqrt.(abs.(ndmrg/2))
vKS_true = [sqrtdens[w] < 1E-12 ? 0. : ((sqrtdens[w+1]-2*sqrtdens[w]+sqrtdens[w-1])/Delta^2)/2/sqrtdens[w]+En/2 for w = 2:Ns-1]

for w = 1:length(vKS_true)
  if vKS_true[w] > 0. || vKS_true[w] < En
    vKS_true[w] = 0.
  end
end

plot!(vcat([0],vKS_true,[0]),label="True KS Potential", init = true)


#=
vKS_true = [sqrtdens[w] < 1E-12 ? 0. : ((sqrtdens[w+1]-2*sqrtdens[w]+sqrtdens[w-1])/Delta^2)/2/sqrtdens[w]+3.25*En/4 for w = 2:Ns-1]

for w = 1:length(vKS_true)
  if vKS_true[w] > 0. || vKS_true[w] < 2*En
    vKS_true[w] = 0.
  end
end

plot(nKS,label="noninteracting density",init=true)

plot!(vKS,label="KS Potential",init=true)

plot!(ndmrg,label="interact density",init=true)

plot!(vext,label="External Potential", init =true)

plot!(gauss_guess,label="initial guess",init=true)

plot!(vcat([0],vKS_true,[0]),label="True KS Potential", init = true)

=#

#=
plot(vKS_true,label="True KS Potential", init =true)
plot!(vKS,label="KS Potential",init=true)
plot!(vext,label="External Potential", init =true)
=#
#400:600]#