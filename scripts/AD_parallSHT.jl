function alm2map(d_alm, d_map, nthreads::Integer)

    alm2map!(d_alm, d_map, nthreads=nthreads)

    return d_map
end

#=function alm2map(d_alm, nside::Integer, comm::MPI.Comm, nthreads::Integer; root=0)

    if MPI.Comm_rank(comm) == root
        h_map = HealpixMap{Float64, RingOrder}(nside)
    else
        h_map = nothing
    end
    d_map = DMap{RR}(comm)
    HealpixMPI.Scatter!(h_map, d_map)

    MPI.Barrier(comm)
    alm2map!(d_alm, d_map, nthreads=nthreads)
    MPI.Barrier(comm)

    return d_map
end=#

function adjoint_alm2map(d_map, d_alm, nthreads::Integer)

    adjoint_alm2map!(d_map, d_alm, nthreads=nthreads)
    MPI.Barrier(d_map.info.comm)

    return d_alm
end

@adjoint function alm2map(d_alm, d_map, nthreads::Integer)
    p = alm2map(d_alm, d_map, nthreads)
    function alm2map_PB(adj_p)
        adj_a = deepcopy(d_alm)
        adjoint_alm2map!(adj_p, adj_a, nthreads=nthreads)
        adj_a *= 2
        for i in 1:(d_alm.info.lmax + 1)
            adj_a.alm[i] /= 2.
        end
        return (adj_a, nothing, nothing)
    end
    return p, alm2map_PB
end

@adjoint function HealpixMPI.DMap{S,T}(pixels::Matrix{T}, info::GeomInfoMPI) where {S<:Strategy, T<:Real}
    map = DMap{S,T}(pixels, info)
    function DMap_PB(adj_map)
        adj_pix = adj_map.pixels
        adj_info = nothing
        return (adj_pix, adj_info)
    end
    return map, DMap_PB
end