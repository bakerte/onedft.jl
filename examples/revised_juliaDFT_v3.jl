
using DSP
import LinearAlgebra
#              +------------+
#>-------------| Parameters |-------------<
#              +------------+

#NOTE:  restart for different parameters

const R = 0	#interatomic distance
const Z = 4	#atomic number
const Na = 1	#number of atoms
const Ne = 4	#number of electrons

const Nup = div(Ne+1,2) 	#number of up electrons
const Ndn = Ne-Nup 		#number of down electrons

const edgedist = 20.   #distance to edge of box

const nlev = 5
const spacing = 1.0
const Delta = spacing * 0.5^nlev		#fine grid spacing

const lam = 0.4 	#DFT mixing parameter

const L=2edgedist+Delta+(Na-1)*R	#total length

const A = 1.071295
const a = 2.385345
const kappa = 1.0/a

const maxsteps=10

const ndim=round(Integer,L/Delta)

function showvar(s)     # Works on a symbol, so:  i=2;  showvar(:i)
    println(s," = ",eval(s))
end
showvar(:R); showvar(:Z); showvar(:Na); showvar(:Ne); showvar(:Nup);
showvar(:Ndn); showvar(:nlev); showvar(:spacing); showvar(:Delta);
showvar(:edgedist); showvar(:L); showvar(:ndim); showvar(:lam);
showvar(:maxsteps)

#using PyPlot

#              +------------------------+
#>-------------|     Initialization     |-------------<
#              +------------------------+

dens=zeros(ndim)
densup=zeros(ndim)
densdn=zeros(ndim)

newdensup=zeros(ndim)
newdensdn=zeros(ndim)
newdens=zeros(ndim)
nextdensup=zeros(ndim)
nextdensdn=zeros(ndim)
nextdens=zeros(ndim)

#exponential interaction
vexp(x::Float64) = A*exp(-kappa * abs(x))

#position of a atom
atom(i::Integer) = Delta+edgedist+(i-1)*R

#external potential
v(x::Float64) = sum(i->-Z*vexp(x-atom(i)),1:Na)

vKSup=zeros(ndim)
vKSdn=zeros(ndim)

showv=Float64[v(i*Delta) for i=1:ndim]

#position for the plot and dens above
pos = Float64[i*Delta for i=1:ndim]

#trial density
const width=.5
dens = Float64[exp(-(Delta*i-(L/2))^2/(2width)) for i=1:ndim]
norm = sum(i->dens[i]*Delta,1:ndim)

dens=dens/norm*Ne
densup=dens*Nup/Ne
densdn=dens*Ndn/Ne

#              +----------------------------------------+
#>-------------| Definitions Hamiltonian & KS potential |-------------<
#              +----------------------------------------+

Hup=zeros(ndim,ndim)
Hdn=zeros(ndim,ndim)

vKSup=zeros(ndim)
vKSdn=zeros(ndim)
vH=zeros(ndim)

include("exchangecorrelation.jl")

#              +----------------------------------------+
#>-------------|       Other DFT Energies               |-------------<
#              +----------------------------------------+

Ts(vec::Vector{Float64}) = sum(i->-vec[i] * (vec[i-1]-2vec[i]+vec[i+1])/(2Delta),2:ndim-1)

const vsmlim = ndim
vsmall = Float64[vexp((k-vsmlim-1)*Delta) for k=1:2*vsmlim+1]

function convolve(x::Vector{Float64})
    res = DSP.conv(x,vsmall)
    Float64[Delta*res[i+vsmlim] for i=1:length(x)]
end

function U(dens::Vector{Float64}) 
#res1 = sum(i->(sum(j->dens[i]*dens[j] * vexp((i-j)*Delta)/2*Delta^2,1:ndim)),1:ndim)
    vH = convolve(dens)
    res2 = LinearAlgebra.dot(dens,vH)*0.5*Delta
#    println("res1, 2 are ",res1, "  ",res2)
    res2
end

#              +----------------+
#>-------------| KS algorithm   |-------------<
#              +----------------+

vecsup = rand(ndim,Nup) * 1.0e-8
vecsdn = rand(ndim,max(1,Ndn)) * 1.0e-8

Hoff = Float64[-0.5/Delta^2 for i=1:ndim-1]
Hdup = Float64[1.0/Delta^2 for i=1:ndim]
Hddn = Float64[1.0/Delta^2 for i=1:ndim]

function run(maxsteps,Delta,dens,densup,densdn)

energylast=0.

for step = 1:maxsteps
#    plot(pos,dens, "b-",pos,showv,"r--")

    println("Starting convolve")#; flush(STDOUT)
    vH = convolve(dens)
    println("Done with convolve")#; flush(STDOUT)

    for i = 1:ndim
        	#vc was defined as d(vc)/dnup--switching the arguments does d(ndn) 
        vKSup[i] = v(i*Delta) + vH[i] + vxup(densup[i],densdn[i]) + vc(densup[i],densdn[i])
        vKSdn[i] = v(i*Delta) + vH[i] + vxdn(densup[i],densdn[i]) + vc(densdn[i],densup[i])
    end
    
    for i = 1:ndim
        Hdup[i] = vKSup[i] + 1/Delta^2
        Hddn[i] = vKSdn[i] + 1/Delta^2
    end

    println("Starting eigs")#; flush(STDOUT)
    SHup = LinearAlgebra.SymTridiagonal(Hdup,Hoff)
    lambdaup = LinearAlgebra.eigvals(SHup,1:Nup)
    vecsup = LinearAlgebra.eigvecs(SHup,lambdaup)
    #println("nconv = ", nconv, ", niter = ",niter, ", nmult = ",nmult)

    if Ndn > 0
	SHdn = LinearAlgebra.SymTridiagonal(Hddn,Hoff)
        lambdadn = LinearAlgebra.eigvals(SHdn,1:Ndn)
	    vecsdn = LinearAlgebra.eigvecs(SHdn,lambdadn)
    end
    println("Done with eigs")#; flush(STDOUT)
    
    	#creation of new density for next iteration
    for n = 1:ndim
        newdensup[n] = sum(m->vecsup[n,m]^2/Delta,1:Nup)	#divide by Delta for normalization of grid
        if (Ndn != 0)
	    newdensdn[n] = sum(m->vecsdn[n,m]^2/Delta,1:Ndn)
        end
    end

    newdens = newdensup + newdensdn

    nextdens = lam*newdens+(1-lam)*dens	#used in next iteration
    nextdensup = lam*newdensup+(1-lam)*densup
    nextdensdn = lam*newdensdn+(1-lam)*densdn
    
    	#update densities for next step
    dens = nextdens
    densup = nextdensup
    densdn = nextdensdn
    
    	#calculate the energy and energy difference from last iteration
    tsenergyup = sum(m->Ts(vecsup[:,m]/sqrt(Delta)),1:Nup)	#sqrt(Delta) for the norm of the wavefunction on the grid
    if (Ndn != 0)
        tsenergydn = sum(m->Ts(vecsdn[:,m]/sqrt(Delta)),1:Ndn)
    else
	tsenergydn = 0
    end
    println("Starting U energy")#; flush(STDOUT)
    ehenergy = U(dens)
    println("Done with U energy")#; flush(STDOUT)
    exenergy = sum(i->ex(densup[i],densdn[i])*Delta,1:ndim)	#exchange energy
    ecenergy = sum(i->ec(densup[i],densdn[i])*Delta,1:ndim)	#correlation
    venergy = sum(i->dens[i]*v(i*Delta)*Delta,1:ndim)		#external potential energy
    energy=tsenergyup+tsenergydn+ehenergy+exenergy+ecenergy+venergy
    println("Energy: ",energy)
    println("Ts[n] = ",tsenergyup+tsenergydn," | Ts[nup] = ",tsenergyup," | Ts[ndn] = ",tsenergydn," | U[n] = ",ehenergy)
    println("Ex[n] = ",exenergy," | Ec[n] = ",ecenergy," | V[n] = ",venergy)
    println("Energy difference: ",energy-energylast)
    abs(energy-energylast) < 1.0e-14 && break
    energylast=energy
#    flush(STDOUT)
end
#=
println("n  v[n]  v[n,j] ")
for n=1:ndim
    print(n," ")
    print(showv[n],"     ")
    for j=1:Nup
	print(vecsup[n,j]," ")
    end
    println(" ")
end
=#
end

run(maxsteps,Delta,dens,densup,densdn)


