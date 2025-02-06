# Manipulation of alm objects and adoint rules
function from_alm_to_healpix_alm(alm, l_max, nbin, comm; root=0)

    Alms = []
    for i in 1:nbin
        alm_array = zeros(ComplexF64, numberOfAlms(l_max))
        for l in 1:l_max+1
            alm_array[l] = alm[l][i, 1]
        end
        j = l_max + 1
        for m in 2:2:(2*l_max + 1)
            for l in (Int(m/2) +1):(l_max+1)
                j += 1
                alm_array[j] = alm[l][i,m] + alm[l][i,m+1]*im
            end
        end
        push!(Alms, Alm(l_max, l_max, alm_array))
    end

    return Alms
end

function from_healpix_alm_to_alm(Alms, lmax, nbin, comm; root=0)

    alm_array = []
    for l in 0:lmax
        alm = Matrix{Float64}(undef, (nbin, 2*(l + 1)))
        for i in 1:nbin
            j = 1
            for m in each_m_idx(Alms[i], l)
                alm[i,j]=real(Alms[i].alm[m])
                alm[i,j+1]=imag(Alms[i].alm[m])
                j+=2
            end
        end
        push!(alm_array, alm[:, 1:end .!=2])
    end

    return alm_array
end

function ChainRulesCore.rrule(::typeof(from_alm_to_healpix_alm), alm, l_max, nbin, comm; root=0)

    y = from_alm_to_healpix_alm(alm, l_max, nbin, comm, root=root)

    function fatha_pullback(ȳ)

        x̄ = @thunk(from_healpix_alm_to_alm(ȳ, l_max, nbin, comm, root=root))

        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, fatha_pullback
end

#=function ChainRulesCore.rrule(::typeof(from_healpix_alm_to_alm), hpix_alm, l_max, nbin)
    y = from_healpix_alm_to_alm(hpix_alm, l_max, nbin)
    function fhata_pullback(ȳ)
        x̄ = @thunk(from_alm_to_healpix_alm(ȳ, l_max, nbin))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, fhata_pullback
end=#

function x_vecmat2vec(x, lmax::Int64, nbin::Int64, comm; root=0)

    all_x_per_field = reduce(hcat, x)
    vec_x = reshape(all_x_per_field, (nbin*(numberOfAlms(lmax)*2 - (lmax+1)), 1))
    vec_x = vec(vec_x)
    
    return vec_x
end

function x_vec2vecmat(vec_x, lmax::Int64, nbin::Int64, comm; root=0)
    
    all_x_per_field = reshape(vec_x, ( nbin, 2*numberOfAlms(lmax)-(lmax+1)))
    x = Vector{Matrix{Float64}}(undef, lmax+1)
    for l in 0:lmax
        j_in = l^2 + 1
        j_fin = j_in + 2*l
        x[l+1] = all_x_per_field[:,j_in:j_fin]
    end

    return x
end

@adjoint function x_vec2vecmat(vec_x, lmax::Int64, nbin::Int64, comm; root=0)

    y = x_vec2vecmat(vec_x, lmax, nbin, comm, root=root)

    function x_vec2vecmat_pullback(ȳ)

        adj_vec_x = x_vecmat2vec(ȳ, lmax, nbin, comm, root=root)

        return (adj_vec_x, nothing, nothing, nothing, nothing)
    end

    return y, x_vec2vecmat_pullback
end

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

function from_alm_to_healpix_alm(alm, l_max, nbin)

    Alms = []
    for i in 1:nbin
        alm_array = zeros(ComplexF64, numberOfAlms(l_max))
        for l in 1:l_max+1
            alm_array[l] = alm[l][i, 1]
        end
        j = l_max + 1
        for m in 2:2:(2*l_max + 1)
            for l in (Int(m/2) +1):(l_max+1)
                j += 1
                alm_array[j] = alm[l][i,m] + alm[l][i,m+1]*im
            end
        end
        push!(Alms, Alm(l_max, l_max, alm_array))
    end

    return Alms
end

function from_healpix_alm_to_alm(Alms, lmax, nbin)

    alm_array = []
    for l in 0:lmax
        alm = Matrix{Float64}(undef, (nbin, 2*(l + 1)))
        for i in 1:nbin
            j = 1
            for m in each_m_idx(Alms[i], l)
                alm[i,j]=real(Alms[i].alm[m])
                alm[i,j+1]=imag(Alms[i].alm[m])
                j+=2
            end
        end
        push!(alm_array, alm[:, 1:end .!=2])
    end

    return alm_array
end

function ChainRulesCore.rrule(::typeof(from_alm_to_healpix_alm), alm, l_max, nbin)

    y = from_alm_to_healpix_alm(alm, l_max, nbin)

    function fatha_pullback(ȳ)

        x̄ = @thunk(from_healpix_alm_to_alm(ȳ, l_max, nbin, comm, root=root))
    
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, fatha_pullback
end

#=function ChainRulesCore.rrule(::typeof(from_healpix_alm_to_alm), hpix_alm, l_max, nbin)
    y = from_healpix_alm_to_alm(hpix_alm, l_max, nbin)
    function fhata_pullback(ȳ)
        x̄ = @thunk(from_alm_to_healpix_alm(ȳ, l_max, nbin))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, fhata_pullback
end=#

function x_vecmat2vec(x, lmax::Int64, nbin::Int64)

    all_x_per_field = reduce(hcat, x)
    vec_x = reshape(all_x_per_field, (nbin*(numberOfAlms(lmax)*2 - (lmax+1)), 1))
    vec_x = vec(vec_x)
    
    return vec_x
end

function x_vec2vecmat(vec_x, lmax::Int64, nbin::Int64)
    
    all_x_per_field = reshape(vec_x, ( nbin, 2*numberOfAlms(lmax)-(lmax+1)))
    x = Vector{Matrix{Float64}}(undef, lmax+1)
    for l in 0:lmax
        j_in = l^2 + 1
        j_fin = j_in + 2*l
        x[l+1] = all_x_per_field[:,j_in:j_fin]
    end

    return x
end

@adjoint function x_vec2vecmat(vec_x, lmax::Int64, nbin::Int64)

    y = x_vec2vecmat(vec_x, lmax, nbin)

    function x_vec2vecmat_pullback(ȳ)

        adj_vec_x = x_vecmat2vec(ȳ, lmax, nbin)
    
        return (adj_vec_x, nothing, nothing)
    end

    return y, x_vec2vecmat_pullback
end