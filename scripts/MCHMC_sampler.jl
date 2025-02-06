using Distributions
using LinearAlgebra
using StatsFuns
using StatsBase

using LogDensityProblems
using LogDensityProblemsAD
using AbstractDifferentiation
using MCMCDiagnosticTools
using AdvancedHMC
using MicroCanonicalHMC
using Pathfinder
using Transducers

using Healpix
using HealpixMPI
using MPI

using Plots
using StatsPlots
using LaTeXStrings

using Random
using ProgressMeter
using BenchmarkTools
using NPZ
using CSV
using DataFrames
using Test

using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore

include("AD_parallSHT.jl")
include("gen_datatest.jl")
include("reparam.jl")
include("utils.jl")
include("inference_model.jl")
include("neglogproblem.jl")

MCHMC_prefix = randstring(5)
init_seed = rand(1:10_000)

seed = 1123
Random.seed!(seed)

#   RESOLUTION PARAMETERS
nside = 512
lmax = 2*nside - 1

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
ncore = 32

#   REALIZATION MAP
realiz_Cl, realiz_HAlm, realiz_HMap = Realization("Capse_fiducial_Dl.csv", nside, lmax, seed)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(realiz_Cl))

d = length(realiz_θ)

#   SURVEY MASK
mask_512 = readMapFromFITS("wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits",2,Float64)
mask_nside = udgrade(nest2ring(mask_512), nside)
for i in 1:length(mask_nside.pixels)
    if mask_nside.pixels[i]<=0.5
        mask_nside.pixels[i]=1
    else
        mask_nside.pixels[i]=0
    end
end

#   GENERATED DATA MEASUREMENTS
#   Noise
ϵ=150
N = ϵ*ones(nside2npix(nside))
N[mask_nside.==1] .= 5*10^4
#   Gaussian beam and pixel window function
Bl = ones(length(realiz_Cl))#gaussbeam(0.001, lmax, pol=false)
Pl = ones(length(realiz_Cl))#pixwin(nside, pol=false)[1:lmax+1]
BP_l = Bl.*Pl

#   Data Map
gen_Cl, gen_HAlm, gen_HMap = Measurement(realiz_Cl, Bl, Pl, mask_nside, N, nside, lmax, seed)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(gen_Cl))
invN_HMap = HealpixMap{Float64,RingOrder}(1 ./ N)

#   STARTING POINT
start_Cl, start_HAlm, start_HMap = StartingPoint(gen_Cl, nside)
start_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([start_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(start_Cl))

#   PROMOTE HEALPIX.MAP TO HEALIPIXMPI.DMAP
gen_DMap = DMap{RR}(comm)
invN_DMap = DMap{RR}(comm)
HealpixMPI.Scatter!(gen_HMap, gen_DMap, comm, clear=true)
HealpixMPI.Scatter!(invN_HMap, invN_DMap, comm, clear=true)

helper_DMap = deepcopy(gen_DMap)

nlp = nℓπ(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, BP_l=Bl.*Pl, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
nlp_grad = nℓπ_grad(start_θ, data=gen_DMap,  helper_DMap=helper_DMap, lmax=lmax, nside=nside, BP_l=Bl.*Pl, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

#=
## BENCHMARKING POSTERIOR and POSTERIOR+GRADIENTS TIMINGS
MPI.Barrier(comm)
nlp_bm = @benchmark nℓπ(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
MPI.Barrier(comm)
nlp_grad_bm = @benchmark nℓπ_grad(start_θ, data=gen_DMap,  helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, ncore=ncore, comm=comm, root=root)
print(mean(nlp_bm).time)
print(mean(nlp_grad_bm).time)
=#

## PATHFINDER INITIALIZATION
#PF_prefix = "jmUzm"
Random.seed!(init_seed)
PF_start_θ = rand(MvNormal(realiz_θ,0.01*I)) #npzread("MPI_chains/$(PF_prefix)_PATHinit_$(nside).npy")[:,end-2]

function MCHMCℓπ(θ)
    return -nℓπ(θ) #ricorda -1
end

function MCHMCℓπ_grad(x)
    f, df = nℓπ(x), nℓπ_grad(x) #ricorda -1
    return -f, -df[1]
end

target = CustomTarget(MCHMCℓπ, MCHMCℓπ_grad, PF_start_θ)

n_adapts, n_steps = 2_000, 10_000
spl = MicroCanonicalHMC.MCHMC(n_adapts, 0.001, integrator="LF", 
            adaptive=true, tune_eps=true, tune_L=false, eps=10.0, tune_sigma=false, L=sqrt(d), sigma=ones(d))

MPI.Barrier(comm)
t0 = time()
samples_MCHMC = Sample(spl, target, n_steps, init_params=PF_start_θ, dialog=true, thinning=2)
MPI.Barrier(comm)
MCHMC_t = time() - t0

npzwrite("MPI_chains/$(MCHMC_prefix)_mask_MCHMC_nside_$nside.npy", samples_MCHMC)

MCHMC_ess, MCHMC_rhat = Summarize(samples_MCHMC')
npzwrite("MPI_chains/$(MCHMC_prefix)_mask_MCHMC_EssRhat_nside_$nside.npy", [MCHMC_ess MCHMC_rhat])
npzwrite("MPI_chains/$(MCHMC_prefix)_mask_MCHMC_SumPerf_nside_$nside.npy", [MCHMC_t mean(MCHMC_ess) median(MCHMC_rhat)])

MPI.Finalize()
