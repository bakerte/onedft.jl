#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#

try 
  path = "../../"
  include(join([path,"DMRjulia.jl"]))
catch
  using DMRJtensor
  using TensorPACK
end

Ns = 101
Delta = 0.1

@makeQNs "fermion" U1 U1
Qlabels = [[fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]]

Ne = Ns
Ne_up = 1 #ceil(Int64,div(Ne,2))
Ne_dn = 1 #Ne-Ne_up
QS = 4
Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

psi = MPS(QS,Ns)
upsites = [i for i = 1:Ne_up]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 1:Ne_dn]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

qpsi = makeqMPS(Qlabels,psi)

mu = 1/Delta^2
#HubU = 4.0
t = -1/(2*Delta^2)

A = 1.071295 #1.0
kappa = 1/2.385345 #0.5

Z = 2

V = [-Z * A*exp(-kappa*Delta*abs(i-cld(Ns-1,2))) for i = 1:Ns] #[(Delta*(i-cld(Ns-1,2)))^2/2 for i = 1:Ns]

function H(i::Int64)
        onsite = mu * Ndens + V[i] * Ndens#+ HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end


#println("Making qMPO")
mpo = makeMPO(H,QS,Ns)

expmpo = expMPO(exp(-Delta*kappa),A*Ndens,Ndens,Ns)
mpo += expmpo

#=
function makeH(Ns)
  mpo = 0
  for i = 1:Ns-1
    mpo += mpoterm(-t,Cup,i,Cup',i+1,F)
    mpo += mpoterm(t,Cup',i,Cup,i+1,F)
    mpo += mpoterm(-t,Cdn,i,Cdn',i+1,F)
    mpo += mpoterm(t,Cdn',i,Cdn,i+1,F)
  end
  for i = 1:Ns
    mpo += mpoterm(mu,Ndens,i)
    mpo += mpoterm(V[i],Ndens,i)
  end
  return MPO(mpo)
end

mpo = makeH(Ns)
for w = 1:length(mpo)
  println(w," ",size(mpo[w]))
end
=#
qmpo = makeqMPO(Qlabels,mpo)


println("#############")
println("QN version")
println("#############")

QNenergy = dmrg(qpsi,qmpo,m=5,sweeps=100,cutoff=1E-9,goal=1E-6)
En = QNenergy

qNup,qNdn,qNdens = Qtens([Qlabels[1],inv.(Qlabels[1])],Nup,Ndn,Ndens)

densities = correlation(qpsi,qNdens)


using Plots
plot(densities)
plot!(V)

####
#### Finding the exact Kohn-Sham potential
####

include("../oneDFT.jl")
vext = V
ndmrg = densities


c = cld(Ns,2)
vKS = copy(vext) #+ 0.01*rand(Ns)
ThomasFermi = vKS

g = 50
gauss_guess = [-4*exp(-Delta/g*abs(i-c)^2/2) for i = 1:Ns]

vKS,nKS = findvKS(ndmrg,Delta,Ne_up,Ne_dn,vKS=copy(vext))


using Plots

plot(nKS,label="noninteracting density",init=true)
plot!(vKS,label="KS Potential",init=true)
plot!(ndmrg,label="interact density",init=true)
plot!(vext,label="External Potential", init =true)
plot!(gauss_guess,label="initial guess",init=true)


#We probably need to smooth it out to actually do the time evolution...
#The following computes the exact Kohn-Sham potential for two electrons

sqrtdens = sqrt.(ndmrg/2)
vKS_true = [sqrtdens[w] < 1E-12 ? 0. : ((sqrtdens[w+1]-2*sqrtdens[w]+sqrtdens[w-1])/Delta^2)/2/sqrtdens[w]+En/2 for w = 2:Ns-1]

for w = 1:length(vKS_true)
  if vKS_true[w] > 0. || vKS_true[w] < En
    vKS_true[w] = 0.
  end
end

plot!(vcat([0],vKS_true,[0]),label="True KS Potential", init = true)

