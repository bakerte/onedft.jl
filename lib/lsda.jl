

#              +------------+
#>-------------| Parameters |-------------<
#              +------------+

#NOTE:  restart for different parameters

R = 0	#interatomic distance
Z = 2	#atomic number
Na = 1	#number of atoms
Ne = 2	#number of electrons

Nup = div(Ne+1,2) 	#number of up electrons
Ndn = Ne-Nup 		#number of down electrons

edgedist = 20.   #distance to edge of box

nlev = 5
spacing = 1.0
Delta = spacing * 0.5^nlev		#fine grid spacing

lam = 0.1 	#DFT mixing parameter

L=2edgedist+Delta+(Na-1)*R	#total length

A = 1.071295
a = 2.385345
kappa = 1.0/a

maxsteps=1000

ndim=round(Integer,L/Delta)

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

#exponential interaction
vexp(x::Float64;A::Float64=A,kappa::Float64=kappa) = A*exp(-kappa * abs(x))

#position of a atom
atom(i::Integer) = Delta+edgedist+(i-1)*R

#external potential
v(x::Float64) = sum(i->-Z*vexp(x-atom(i)),1:Na)



include("XCEnergy.jl")
import DSP
import LinearAlgebra
import Printf

Ts(vec::Vector{Float64}) = sum(i->-vec[i] * (vec[i-1]-2vec[i]+vec[i+1])/(2Delta),2:ndim-1)


function convolve(x::Vector{W},vsmall::Vector{W}) where W <: Number
    res = DSP.conv(x,vsmall)
    Ns = length(x)
    return res[1+Ns:2Ns] #W[Delta*res[i+Ns] for i=1:length(x)]
end

function solvespindensity(Ns,Nup,H0,vext,vH,densup,densdn)
  vKSup = vext + vH
  @simd for i = 1:Ns
    #vc was defined as d(vc)/dnup--switching the arguments does d(ndn) 
    @inbounds vKSup[i] += vxpol(densup[i])
  end
  @simd for i = 1:Ns
    @inbounds vKSup[i] += vc(densup[i],densdn[i])
  end
  SHup = H0 + LinearAlgebra.Diagonal(vKSup)
  lambdaup = LinearAlgebra.eigvals(SHup,1:Nup)
  vecsup = LinearAlgebra.eigvecs(SHup,lambdaup)

  #creation of new density for next iteration
  newdensup = vecsup[:,1] .^2
  if size(vecsup,2) > 1
    for n = 1:Ns
      @simd for m = 1:Nup
        @inbounds newdensup[n] = vecsup[n,m]^2	#divide by Delta for normalization of grid
      end
    end
  end
  nextdensup = (lam/Delta)*newdensup
  nextdensup += (1-lam)*densup
  densup = nextdensup

  #calculate the energy and energy difference from last iteration
  tsenergyup = 0.
  @simd for m = 1:Nup
    @inbounds psi = vecsup[:,m]
    @inbounds tsenergyup += LinearAlgebra.dot(psi,H0,psi)
  end
  return densup,tsenergyup
end

function addXC(dens,vKS,densup,densdn)
  @simd for i = 1:Ns
    #vc was defined as d(vc)/dnup--switching the arguments does d(ndn) 
    @inbounds vKS[i] += vxpol(densup[i])
  end
  @simd for i = 1:Ns
    @inbounds vKS[i] += vc(densup[i],densdn[i])
  end
  return vKS
end

function solveKS(Ns,Nup,H0,vext,vH,densup,densdn)
  vKSup = vext + vH
  vKSup = addXC(dens,vKSup,densup,densdn)

  SHup = H0 + LinearAlgebra.Diagonal(vKSup)
  lambdaup = LinearAlgebra.eigvals(SHup,1:Nup)
  vecsup = LinearAlgebra.eigvecs(SHup,lambdaup)
  return vecsup
end

function computedensity(vecsup,Delta;Ne=size(vecsup,2))
  #creation of new density for next iteration
  newdensup = [vecsup[w,1]^2/Delta for w = 1:size(vecs,1)]
  if size(vecsup,2) > 1
    for n = 1:Ns
      @simd for m = 1:Nup
        @inbounds newdensup[n] = vecsup[n,m]^2	#divide by Delta for normalization of grid
      end
    end
  end
  return newdensup
end


function mixdensity(newdensup,densup,lam)
  nextdensup = lam*newdensup
  nextdensup += (1-lam)*densup
  densup = nextdensup
  return densup
end

function computeTs(vecsup,H0;Ne=size(vecsup,2))
  #calculate the energy and energy difference from last iteration
  tsenergyup = 0.
  @simd for m = 1:Nup
    @inbounds psi = vecsup[:,m]
    @inbounds tsenergyup += LinearAlgebra.dot(psi,H0,psi)
  end
  return tsenergyup
end

function computeX(densup,densdn;XCfct::Function=ex)
  exenergy = 0.
  @simd for i = 1:length(densup)
    @inbounds exenergy += XCfct(densup[i],densdn[i])
  end
  return exenergy
end

function computeC(densup,densdn)
  return computeX(densup,densdn,XCfct=ec)
end




function computedensity(vecsup,Delta;Ne=size(vecsup,2),Ns=size(vecsup,1))
  #creation of new density for next iteration
  newdensup = vecsup[:,1] .^2
  if Ne > 1
    for n = 1:Ns
      @simd for m = 1:Ne
        @inbounds newdensup[n] = vecsup[n,m]^2	#divide by Delta for normalization of grid
      end
    end
  end
  return newdensup/Delta
end

function computeTs(vecsup,H0;Ne=size(vecsup,2))
  #calculate the energy and energy difference from last iteration
  tsenergyup = 0.
  @simd for m = 1:Ne
    @inbounds psi = vecsup[:,m]
    @inbounds tsenergyup += LinearAlgebra.dot(psi,H0,psi)
  end
  return tsenergyup
end

#=
function mixKS(dens::Array{W,1},vext::Array{W,1},Nup::Integer,Ndn::Integer) where W <: Number

end

function mixKS(densup::Array{W,1},densdn::Array{W,1},vext::Array{W,1},Nup::Integer,Ndn::Integer) where W <: Number

end
=#

function mixKS(dens::Array{W,1},vext::Array{W,1},Nup::Integer,Ndn::Integer;restricted::Bool=false) where W <: Number

  Ns = ndim
  Ne = Nup+Ndn




  #  vsmlim = ndim
  vsmall = W[vexp((k-Ns-1)*Delta) for k=1:2*Ns+1]

  energylast=100000.

  densnorm = sum(dens)*Delta

  if !isapprox(densnorm,Ne)
    dens_renorm = Ne/densnorm
    @simd for w = 1:Ns
      @inbounds dens[w] *= dens_renorm
    end
  end
  densup=dens*Nup/Ne
  densdn=dens*Ndn/Ne











  Hoff = -ones(W,Ns-1)/Delta^2/2
  Hdup = ones(W,Ns)/Delta^2

  H0 = LinearAlgebra.SymTridiagonal(Hdup,Hoff)
  vH = convolve(dens,vsmall)*Delta

  for step = 1:maxsteps
    t1 = time()
    

    if restricted
      #=
      @simd for w = 1:Ns
        @inbounds vH[w] += vxpol(densup[w])
      end
      =#
      vecsup = solveKS(Ns,Nup,H0,vext,vH,densup,densdn)
      newdensup = computedensity(vecsup,Delta,Ne=Nup)
      densup = mixdensity(newdensup,densup,lam)
      tsenergyup = computeTs(vecsup,H0,Ne=Nup)

      if Ndn != Nup
#        densup,tsenergyup = solvespindensity(Ns,Nup,H0,vext,vH,densup,densdn)
        newdensdn = computedensity(vecsup,Delta,Ne=Ndn)
        densdn = mixdensity(newdensdn,densdn,lam)
        tsenergydn = computeTs(vecsdn,H0,Ne=Ndn)
      else
        densdn,tsenergydn = densup,tsenergyup
      end
    else
      if Nup > 0
#        densup,tsenergyup = solvespindensity(Ns,Nup,H0,vext,vH,densup,densdn)
        vecsup = solveKS(Ns,Nup,H0,vext,vH,densup,densdn)
        newdensup = computedensity(vecsup,Delta,Ne=Nup)
        densup = mixdensity(newdensup,densup,lam)
        tsenergyup = computeTs(vecsup,H0,Ne=Nup)
      else
        tsenergyup = 0.
      end
      if Ndn > 0
#        densdn,tsenergydn = solvespindensity(Ns,Ndn,H0,vext,vH,densdn,densup)
        vecsdn = solveKS(Ns,Ndn,H0,vext,vH,densdn,densup)
        newdensdn = computedensity(vecsdn,Delta,Ne=Ndn)
        densdn = mixdensity(newdensdn,densdn,lam)
        tsenergydn = computeTs(vecsdn,H0,Ne=Ndn)
      else
        tsenergydn = 0.
      end
    end


    #update densities for next step
    dens = densup + densdn #nextdens
    

    vH = convolve(dens,vsmall)*Delta
    Uenergy = LinearAlgebra.dot(dens,vH)
    Uenergy *= Delta/2

    exenergy = computeX(densup,densdn)*Delta	#exchange energy
    ecenergy = computeC(densup,densdn)*Delta	#correlation

    venergy = LinearAlgebra.dot(dens,vext)*Delta		#external potential energy
    energy=tsenergyup+tsenergydn+Uenergy+exenergy+ecenergy+venergy
    t2 = time()
    println("iteration: $step (",Printf.@sprintf("%.3f",(t2-t1)*1E3),"ms)")
    println("Energy: ",energy)
    println("Ts[n] = ",tsenergyup+tsenergydn," | Ts[nup] = ",tsenergyup," | Ts[ndn] = ",tsenergydn," | U[n] = ",Uenergy)
    println("Ex[n] = ",exenergy," | Ec[n] = ",ecenergy," | V[n] = ",venergy)
    println("Energy difference: ",energy-energylast)
  #  abs(energy-energylast) < 1.0e-14 && break
    energylast=energy

    println()
  end
  return densup,densdn
end


Ns = ndim
vext = [v(i*Delta) for i = 1:Ns]
#trial density

width=.5
dens = Float64[exp(-(Delta*i-(L/2))^2/(2width)) for i=1:Ns]

densup,densdn = mixKS(dens,vext,Nup,Ndn)



#plot(densup+densdn)



function invertdensity(density::Array{W,1}) where W <: Number
  rho_scr = rand(length(density))
  rho_scr /= sum(rho_scr)
  rho_scr *= sum(density)
  return invertdensity(rho_scr,density)
end



function invertdensity(rho_scr::Array{W,1},density::Array{W,1},Nup::Integer,Ndn::Integer,epsilon::W=0.01;makepotential::Function=nonInt,dU::Number=1E-3,interaction::Function=nonInt) where W <: Number

  Ns = length(density)
  vsmall = W[vexp((k-Ns-1)*Delta) for k=1:2*Ns+1]
  Uenergy = computeU(dens,vsmall,Delta)
  prevU = 1000000.



  Ne = Nup + Ndn
  scdens = density
  densnorm = sum(scdens)
  densup=scdens*Nup/densnorm
  densdn=scdens*Ndn/densnorm


  vKSup = zeros(Ns)
  vKSdn = zeros(Ns)


  while abs(Uenergy - prevU) > dUtol

    vH = convolve(dens,vsmall)*Delta
    vKS = vext + vH
    vKSup = addXC(dens,vKS,densup,densdn)
    vKSdn = addXC(dens,vKS,densdn,densup)

    KSenergiesup,vecsup = solveKS(vKSup,Nex=Nup)
    KSenergiesdn,vecsdn = solveKS(vKSdn,Nex=Ndn)

    newdensup = computedensity(vecsup,Delta,Ne=Nup)
    newdensdn = computedensity(vecsdn,Delta,Ne=Ndn)

    newdensity = newdensup + newdensdn

    scdens += epsilon*(newdensity - density)
    densnorm = sum(scdens)
    scdens = scdens/densnorm
    densup=scdens*Nup
    densdn=scdens*Ndn
    

    prevU = Uenergy
    Uenergy = computeU(scdens,vsmall,Delta)
  end
  return vKSup+vKSdn
end
export invertdensity





