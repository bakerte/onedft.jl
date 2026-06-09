

include("../oneDFT.jl")

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

const maxsteps = 100

const ndim = round(Integer,L/Delta)

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


#Hup=zeros(ndim,ndim)
#Hdn=zeros(ndim,ndim)

#trial density
const width = 0.5
dens = Float64[exp(-(Delta*i-(L/2))^2/(2width)) for i=1:ndim]
norm = sum(i->dens[i]*Delta,1:ndim)

dens=dens/norm*Ne
#densup=dens*Nup/Ne
#densdn=dens*Ndn/Ne



vext = [oneDFT.v(i*Delta,Z,Na,Delta,edgedist,R) for i = 1:length(dens)]


En,vKS,newdens,orbup,orbdn = solveKS(vext,dens,Delta,#=densup,densdn,=#Nup,Ndn,maxiter=maxsteps,mix=lam)


using Plots
plot(vext)
plot!(vKS)
plot!(newdens)


#vKSup=zeros(ndim)
#vKSdn=zeros(ndim)

#showv=Float64[v(i*Delta) for i=1:ndim]

#position for the plot and dens above
#pos = Float64[i*Delta for i=1:ndim]


