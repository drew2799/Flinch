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

HMC_prefix = randstring(5)
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
Bl = gaussbeam(0.001, lmax, pol=false)
Pl = pixwin(nside, pol=false)[1:lmax+1]
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
PF_start_θ = rand(MvNormal(realiz_θ,0.01*I)) #npzread("MPI_chains/$(PF_prefix)_PATHinit_$(nside).npy")[:,end-8]

struct LogTargetDensity
    dim::Int
end

LogDensityProblemsAD.logdensity(p::LogTargetDensity, θ) = -nℓπ(θ)
LogDensityProblemsAD.dimension(p::LogTargetDensity) = p.dim
LogDensityProblemsAD.capabilities(::Type{LogTargetDensity}) = LogDensityProblemsAD.LogDensityOrder{1}()

ℓπ = LogTargetDensity(d)
n_LF = 50
n_samples, n_adapts = 11_000, 10_000

metric = DiagEuclideanMetric(d)
ham = Hamiltonian(metric, ℓπ, Zygote)
initial_ϵ = find_good_stepsize(ham, PF_start_θ) #0.1
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_LF)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

MPI.Barrier(comm)
t0 = time()
samples_HMC, stats_HMC = sample(ham, kernel, PF_start_θ, n_samples, adaptor, n_adapts; progress=true, verbose=true, drop_warmup = true)
MPI.Barrier(comm)
HMC_t = time() - t0

chain_path = ""

npzwrite("$(chain_path)/$(HMC_prefix)_mask_HMC_nside_$nside.npy", reduce(hcat, samples_HMC))
npzwrite("$(chain_path)/$(HMC_prefix)_mask_HMC_stats_nside_$nside.npy", [[stats_HMC[i][:log_density] for i in 1:1_000] [stats_HMC[i][:hamiltonian_energy] for i in 1:1_000]])

HMC_ess, HMC_rhat = Summarize(samples_HMC)
npzwrite("$(chain_path)/$(HMC_prefix)_mask_HMC_EssRhat_nside_$nside.npy", [HMC_ess HMC_rhat])
npzwrite("$(chain_path)/$(HMC_prefix)_mask_HMC_SumPerf_nside_$nside.npy", [HMC_t mean(HMC_ess) median(HMC_rhat)])

MPI.Finalize()
