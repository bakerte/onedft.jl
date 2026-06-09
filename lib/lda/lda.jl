

#exponential interaction
"""
  vexp(x)

Computes the exponential interaction `A`*exp(-`kappa`*|`x`|)
"""
vexp(x::Float64,A,kappa) = A*exp(-kappa * abs(x))


#position of a atom
"""
    atom(i,Delta,edgedist,R)

computes the position of an atom at the `i`th nucleus with interatomic distance `R`, distance to the edge of the box `edgedist` and grid spacing `Delta`
"""
atom(i::Integer,Delta,edgedist,R) = Delta+edgedist+(i-1)*R

#external potential
"""
    vext(x,Z,Na,Delta,edgedist,R)

compute the external potential at a position `x` with atomic number `Z` number of atoms `Na`, grid spacing `Delta`, distance to the edge of the box `edgedist`, and interatomic distance `R`
"""
function v(x::Float64,Z,Na,Delta,edgedist,R)
  vout = 0
  for i = 1:Na
    vout += -Z*vexp(x-atom(i,Delta,edgedist,R),A,kappa)
  end
  return vout
end


#              +----------------------------------------+
#>-------------| Definitions Hamiltonian & KS potential |-------------<
#              +----------------------------------------+




#include("exchangecorrelation.jl")

#              +----------------------------------------+
#>-------------|       Other DFT Energies               |-------------<
#              +----------------------------------------+

"""
   Ts(vec,Delta)

Computes the kinetic energy functions Ts[n] of an input orbital `vec`, grid spacing `Delta`
"""
function Ts(vec::Vector{Float64},Delta)
  ts = 0
#  newvec = Array{Float64,1}(undef,ndim)
  @inbounds @simd for w = 2:length(vec)-1
    ts += -vec[w] * (vec[w-1] - 2*vec[w] + vec[w+1])/(2*Delta)
  end
  return ts
end

#Ts(vec::Vector{Float64},Delta,ndim) = sum(i->-vec[i] * (vec[i-1]-2vec[i]+vec[i+1])/(2Delta),2:ndim-1)

#const vsmlim = ndim

"""
    convolve(dens,interact,Delta)

Convolves an input vector `dens` with an interaction `interact` and grid spacing `Delta`
"""
function convolve(x::Vector{Float64},vsmall,Delta)

  #TAKE THIS OUT LATER...DID NOT WORK WITH SIZE OF OUTPUTS
  vsmlim = length(x)
#  test_vsmall = Float64[vexp((k-vsmlim-1)*Delta,A,kappa) for k=1:2*vsmlim+1]

    res = DSP.conv(x,vsmall)
#    vsmlim = length(test_vsmall)
    Float64[Delta*res[i+vsmlim] for i=1:length(x)]
end

"""
    U(dens,interact,Delta)

Computes the Hartree energy of an input density `dens`, interaction `interact`, and grid spacing `Delta`

See also: (`convolve`)[@ref]
"""
function U(dens::Vector{Float64},vsmall,Delta) 
#res1 = sum(i->(sum(j->dens[i]*dens[j] * vexp((i-j)*Delta)/2*Delta^2,1:ndim)),1:ndim)
    vH = convolve(dens,vsmall,Delta)
    res2 = LinearAlgebra.dot(dens,vH)*0.5*Delta
#    println("res1, 2 are ",res1, "  ",res2)
    res2
end

#              +----------------+
#>-------------| KS algorithm   |-------------<
#              +----------------+


"""
    solveKS()

Finds the energy, Kohn-Sham potential, and orbitals for an input guess 
"""
function solveKS(v,dens,Delta,Nup,Ndn;Ne=Nup+Ndn,densup=dens*Nup/Ne,densdn=dens*Ndn/Ne,maxiter=100,mix=0.4)

  vsmlim = length(dens)
  vsmall = Float64[vexp((k-vsmlim-1)*Delta,A,kappa) for k=1:2*vsmlim+1]

  ndim = length(dens)

  vecsup = rand(ndim,Nup) * 1.0e-8
  vecsdn = rand(ndim,max(1,Ndn)) * 1.0e-8

  Hoff = Float64[-0.5/Delta^2 for i=1:ndim-1]
  Hdup = Float64[1.0/Delta^2 for i=1:ndim]
  Hddn = Float64[1.0/Delta^2 for i=1:ndim]


  energylast=0.

  newdensup=zeros(ndim)
  newdensdn=zeros(ndim)
#  newdens=zeros(ndim)
  nextdensup=zeros(ndim)
  nextdensdn=zeros(ndim)
  nextdens=zeros(ndim)

  vKSup=zeros(ndim)
  vKSdn=zeros(ndim)
  vH=zeros(ndim)


  @time for step = 1:maxiter
#    plot(pos,dens, "b-",pos,showv,"r--")

#    println("Starting convolve")#; flush(STDOUT)
    vH = convolve(dens,vsmall,Delta)
#    println("Done with convolve")#; flush(STDOUT)

    for i = 1:ndim
        #vc was defined as d(vc)/dnup--switching the arguments does d(ndn) 
        vKSup[i] = v[i] + vH[i] + vxup(densup[i],densdn[i]) + vc(densup[i],densdn[i])
        vKSdn[i] = v[i] + vH[i] + vxdn(densup[i],densdn[i]) + vc(densdn[i],densup[i])
    end
    
    for i = 1:ndim
      Hdup[i] = vKSup[i] + 1/Delta^2
      Hddn[i] = vKSdn[i] + 1/Delta^2
    end

#    println("Starting eigs")#; flush(STDOUT)
    if Nup > 0
      SHup = LinearAlgebra.SymTridiagonal(Hdup,Hoff)
      lambdaup = LinearAlgebra.eigvals(SHup,1:Nup)
      vecsup = LinearAlgebra.eigvecs(SHup,lambdaup)
    end
    #println("nconv = ", nconv, ", niter = ",niter, ", nmult = ",nmult)

    if Ndn > 0
	    SHdn = LinearAlgebra.SymTridiagonal(Hddn,Hoff)
      lambdadn = LinearAlgebra.eigvals(SHdn,1:Ndn)
	    vecsdn = LinearAlgebra.eigvecs(SHdn,lambdadn)
    end
#    println("Done with eigs")#; flush(STDOUT)
    
    	#creation of new density for next iteration
    for n = 1:ndim
      newdensup[n] = 0
      @inbounds @simd for w = 1:Nup
        newdensup[n] += vecsup[n,w]^2/Delta	#divide by Delta for normalization of grid
      end

      newdensdn[n] = 0
      @inbounds @simd for w = 1:Ndn
        newdensdn[n] += vecsdn[n,w]^2/Delta
      end
    end

    newdens = newdensup + newdensdn

    nextdens = mix*newdens+(1-mix)*dens	#used in next iteration
    nextdensup = mix*newdensup+(1-mix)*densup
    nextdensdn = mix*newdensdn+(1-mix)*densdn
    
    	#update densities for next step
    dens = nextdens
    densup = nextdensup
    densdn = nextdensdn
    
    #calculate the energy and energy difference from last iteration
  	tsenergyup = 0
    @inbounds @simd for m = 1:Nup
      tsenergyup += Ts(vecsup[:,m]/sqrt(Delta),Delta)
    end

#    tsenergyup = sum(m->Ts(vecsup[:,m]/sqrt(Delta),Delta,ndim),1:Nup)	#sqrt(Delta) for the norm of the wavefunction on the grid

  	tsenergydn = 0
    @inbounds @simd for m = 1:Ndn
      tsenergydn += Ts(vecsdn[:,m]/sqrt(Delta),Delta)
    end

#    println("Starting U energy")#; flush(STDOUT)
    ehenergy = U(dens,vsmall,Delta)
#    println("Done with U energy")#; flush(STDOUT)
    exenergy = 0
    @inbounds @simd for w = 1:ndim
      exenergy += ex(densup[w],densdn[w])
    end
    exenergy *= Delta	#exchange energy

    ecenergy = 0
    @inbounds @simd for w = 1:ndim
      ecenergy += ec(densup[w],densdn[w])
    end
    ecenergy *= Delta	#correlation

    venergy = 0
    @inbounds @simd for i = 1:ndim
      venergy += dens[i]*v[i] #external potential energy
    end
    venergy *= Delta

#    venergy = sum(i->dens[i]*v(i*Delta,Z,Na,Delta,edgedist,R)*Delta,1:ndim)		

    energy = tsenergyup+tsenergydn+ehenergy+exenergy+ecenergy+venergy
    
    println()
    println("Iteration: $step")
    println("Energy: ",energy)
    println("Ts[n] = ",tsenergyup+tsenergydn," | Ts[nup] = ",tsenergyup," | Ts[ndn] = ",tsenergydn," | U[n] = ",ehenergy)
    println("Ex[n] = ",exenergy," | Ec[n] = ",ecenergy," | V[n] = ",venergy)
    println("Energy difference: ",energy-energylast)

    abs(energy-energylast) < 1.0e-14 && break
    
    energylast=energy
  end

  vKS = vKSup + vKSdn
  orbitalup = vecsup
  orbitaldn = vecsdn

  return energylast,vKS,dens,orbitalup,orbitaldn
end