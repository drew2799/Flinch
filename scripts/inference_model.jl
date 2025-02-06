function NegLogLikelihood(sample_DAlm, helper_DMap, ncore, data, invN, BP_l)
    
    sample_DMap = alm2map(almxfl(sample_DAlm, BP_l), helper_DMap, ncore)
    #sample_DMap = alm2map(sample_DAlm, helper_DMap, ncore)

    REsample_DMap = DMapReparam(sample_DMap, invN)
    REgen_DMap = DMapReparam(data, invN)

    diff_map = DMap_diff(REgen_DMap, REsample_DMap)
    #loglike = NegLogStNorm(diff_map)
    cond = invN.pixels .> 1e-4
    unmask_pixels = DMap_condslice(diff_map, cond)
    #diff_map = DMap_diff(data, sample_DMap)
    loglike = 0.5*unmask_pixels'*unmask_pixels

    return loglike
end

function AlmPrior(sample_alm, sample_Kl, comm; lmax=lmax, root=0)
    
    sample_Cl = Kl2Cl(sample_Kl, comm, root=root)
    
    p_alm0 = 0.5*sum(([sample_alm[l][1,1] for l in 1:lmax+1].^2)./sample_Cl) + 0.5*sum(log.(sample_Cl))
    p_alm = 0.
    for l in 1:lmax+1
        p_alm += (sum((sample_alm[l][:,2:end].^2)./sample_Cl[l]) + (l-1)*log(sample_Cl[l]/2.))
    end
    p = p_alm0 + p_alm

    return p
end

@adjoint function AlmPrior(sample_alm, sample_Kl, comm; lmax=lmax, root=0)
    
    y = AlmPrior(sample_alm, sample_Kl, comm, lmax=lmax, root=root)
    
    function AlmPrior_PB(ȳ)

        sample_Cl = Kl2Cl(sample_Kl, comm, root=root)

        adj_sample_alm = deepcopy(sample_alm)
        for l in 1:lmax+1
            adj_sample_alm[l][1,1] = ȳ * sample_alm[l][1,1]/(sample_Cl[l])
            adj_sample_alm[l][:,2:end] = (2*ȳ/(sample_Cl[l])) .* sample_alm[l][:,2:end]
        end

        adj_sample_Kl = zeros(lmax+1)
        for l in 1:lmax+1
            Al = (sample_alm[l][1,1]^2)/2 + sum(sample_alm[l][:,2:end].^2)
            adj_sample_Kl[l] = ((l-0.5)/sample_Cl[l] - Al/(sample_Cl[l]^2))*1_500*pdf(Normal(0,1), sample_Kl[l])*ȳ
        end
        
        return (adj_sample_alm, adj_sample_Kl, nothing, nothing, nothing)
    end
    return y, AlmPrior_PB
end

function KlPrior(sample_Kl, comm; root=0)

    pKl = 0.5*sum(sample_Kl.^2)

    return pKl
end

@adjoint function KlPrior(sample_Kl, comm; root=0)
    
    pKl = KlPrior(sample_Kl, comm, root=root)

    function KlPrior_PB(adj_pKl)

        adj_Kl = adj_pKl .* sample_Kl

        return (adj_Kl, nothing, nothing)
    end

    return pKl, KlPrior_PB
end

function NegLogPosterior(θ, comm, ncore, helper_DMap; data = gen_DMap, lmax=lmax, nside=nside, invN = invN_DMap, BP_l=BP_l, root = 0)

    θ_length = length(θ)

    #alm = MPIvec_slice(θ, 1, θ_length-lmax-1, comm; root=root) 
    alm = θ[1:θ_length-lmax-1]
    sample_alm = x_vec2vecmat(alm, lmax, 1, comm, root=root)
    #sample_HAlm = MPIAlmvec_getindex(from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root), 1, comm; root=root)
    sample_HAlm = from_alm_to_healpix_alm(sample_alm, lmax, 1, comm, root=root)[1]

    #sample_Kl = MPIvec_slice(θ, θ_length-lmax, θ_length, comm; root=root)
    sample_Kl = θ[θ_length-lmax:θ_length]

    sample_DAlm = HAlm2DAlm(sample_HAlm, comm; clear=true, root=root)

    l = NegLogLikelihood(sample_DAlm, helper_DMap, ncore, data, invN, BP_l)
    p_alm = AlmPrior(sample_alm, sample_Kl, comm, lmax=lmax, root=root)
    p_kl = KlPrior(sample_Kl, comm, root=root)

    nlpost = l+p_alm+p_kl

    return nlpost
end

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
@adjoint function almxfl(DAlm, fl)
    DWlm = almxfl(DAlm, fl)
    function almxfl_PB(adj_DWlm)
        adj_DAlm = almxfl(adj_DWlm, fl)
        return (adj_DAlm, nothing)
    end
    return DWlm, almxfl_PB
end

function DMap_condslice(dmap, bool_cond)
    return dmap.pixels[bool_cond]
end
@adjoint function DMap_condslice(dmap, bool_cond)
    v = DMap_condslice(dmap, bool_cond)
    function DMap_condslice_PB(adj_v)
        adj_dmap = deepcopy(dmap)
        adj_dmap.pixels .= 0.
        adj_dmap.pixels[bool_cond] = adj_v
        return (adj_dmap, nothing)
    end
    return v, DMap_condslice_PB
end

function MPIvec_length(v, comm; root=0)
    if MPI.Comm_rank(comm) == root
        l = length(v)
    else
        l = 0
    end
    return l
end

function MPIAlmvec_getindex(vec_alm, n, comm; root=0)
    if MPI.Comm_rank(comm) == root
        vec_alm_n = vec_alm[n]
    else
        vec_alm_n = nothing
    end
    return vec_alm_n
end
@adjoint function MPIAlmvec_getindex(vec_alm, n, comm; root=0)
    v_n = MPIAlmvec_getindex(vec_alm, n, comm; root=root)
    function MPIAlmvec_getindex_PB(adj_v_n)
        if MPI.Comm_rank(comm) == root
            adj_v = Vector(undef, length(vec_alm))
            for i in 1:length(vec_alm)
                if i == n
                    adj_v[i] = adj_v_n
                else
                    adj_v[i] = Alm(vec_alm[i].lmax, vec_alm[i].lmax, zeros(ComplexF64, length(vec_alm[i])))
                end
            end
        else
            adj_v = nothing
        end
        return (adj_v, nothing, nothing, nothing)
    end
    return v_n, MPIAlmvec_getindex_PB
end

function MPIvec_slice(v, start_n::Int, end_n::Int, comm; root=0)
    if MPI.Comm_rank(comm) == root
        sliced_v = v[start_n:end_n]
    else
        sliced_v = nothing
    end
    return sliced_v
end
@adjoint function MPIvec_slice(v, start_n::Int, end_n::Int, comm; root=0)
    sliced_v = MPIvec_slice(v, start_n, end_n, comm; root=root)
    function MPIvec_slice_PB(adj_sliced_v)
        if MPI.Comm_rank(comm) == root
            adj_v = zeros(length(v))
            adj_v[start_n:end_n] = adj_sliced_v
        else
            adj_v = nothing
        end
        return (adj_v, nothing, nothing, nothing, nothing)
    end
    return sliced_v, MPIvec_slice_PB
end


function HAlm2DAlm(HAlm_i, comm; clear=false, root=0)
    DAlm_i = DAlm{RR}(comm)
    HealpixMPI.Scatter!(HAlm_i, DAlm_i, comm, clear=clear)
    return DAlm_i
end

@adjoint function HAlm2DAlm(HAlm_i, comm; clear=false, root=0)
    DAlm_i = HAlm2DAlm(HAlm_i, comm, clear=clear)
    function HAlm2DAlm_PB(adj_DAlm_i)
        if MPI.Comm_rank(comm) == root
            adj_HAlm_i = deepcopy(HAlm_i)
        else
            adj_HAlm_i = nothing
        end
        HealpixMPI.Gather!(adj_DAlm_i, adj_HAlm_i)
        return (adj_HAlm_i, nothing, nothing, nothing)
    end
    return DAlm_i, HAlm2DAlm_PB
end

# !!! ADJOINT RULE NOT WORKING AS IT SHOULD !!!
#=
function DMap2HMap(DMap_i, comm; clear=false, root=0)
    NSIDE = DMap_i.info.nside
    if MPI.Comm_rank(comm) == root
        HMap_i = HealpixMap{Float64, RingOrder}(NSIDE)
    else
        HMap_i = nothing
    end
    HealpixMPI.Gather!(DMap_i, HMap_i)
    return HMap_i
end

@adjoint function DMap2HMap(DMap_i, comm; clear=false, root=0)
    hmap = DMap2HMap(DMap_i, comm, clear=clear, root=root)
    function DMap2HMap_PB(adj_HMap_i)
        adj_DMap_i = DMap{RR}(comm)
        HealpixMPI.Scatter!(adj_HMap_i, adj_DMap_i, clear=clear)
        return (adj_DMap_i, nothing, nothing, nothing)
    end
    return hmap, DMap2HMap_PB
end
=#

function NegLogStNorm(dmap)
    nln = 0.5*sum2_dmap(dmap)
    return nln
end

function NegLogNorm(dmap, invN)
    tot_map = 0.5*invN*dmap*dmap
    local_s = sum(tot_map.pixels[:,1])
    nln = MPI.Allreduce(local_s, +, dmap.info.comm)
    return nln
end
@adjoint function NegLogNorm(dmap, invN)
    nln = NegLogNorm(dmap, invN)
    function NegLogNorm_PB(adj_nln)
        adj_map = deepcopy(dmap) * invN * (adj_nln)
        return (adj_map, nothing)
    end
    return nln, NegLogNorm_PB
end

function DMap_diff(A, B)
    return A-B
end
@adjoint function DMap_diff(A, B)
    C = DMap_diff(A, B)
    function DMap_diff_PB(adj_C)
        adj_A = deepcopy(adj_C)
        adj_B = -1 * deepcopy(adj_C)
        return (adj_A, adj_B, nothing, nothing)
    end
    return C, DMap_diff_PB
end

function HMap_diff(A, B, comm; root=0)
    if MPI.Comm_rank(comm) == root
        C = HealpixMap{Float64, RingOrder}(A.pixels - B.pixels)
    else
        C = nothing
    end
    return C
end
@adjoint function HMap_diff(A, B, comm; root=0)
    C = HMap_diff(A, B, comm, root=root)
    function HMap_diff_PB(adj_C)
        if MPI.Comm_rank(comm) == root
            adj_A = deepcopy(adj_C)
            adj_B = -1 * deepcopy(adj_C)
        else
            adj_A = nothing
            adj_B = nothing
        end
        return (adj_A, adj_B, nothing, nothing)
    end
    return C, HMap_diff_PB
end

function sum2_dmap(d_map)
    tot_map = d_map * d_map
    local_s = sum(tot_map.pixels[:,1])
    global_s = MPI.Allreduce(local_s, +, d_map.info.comm)
    return global_s
end
@adjoint function sum2_dmap(d_map)
    s = sum2_dmap(d_map)
    function sum2_dmap_PB(adj_s)
        adj_map = deepcopy(d_map) * (2*adj_s)
        return (adj_map,)
    end
    return s, sum2_dmap_PB
end

function sum2_hmap(h_map, comm; root=0)
    if MPI.Comm_rank(comm) == root
        s2 = sum(h_map.pixels .^ 2)
    else
        s2 = 0
    end
    return s2
end
@adjoint function sum2_hmap(h_map, comm; root=0)
    s2 = sum2_hmap(h_map, comm, root=root)
    function sum2_hmap_PB(adj_s2)
        if MPI.Comm_rank(comm) == root
            adj_map = HealpixMap{Float64, RingOrder}(deepcopy(h_map.pixels) .* 2*adj_s2)
        else
            adj_map = nothing
        end
        return (adj_map, nothing, nothing)
    end
    return s2, sum2_hmap_PB
end
