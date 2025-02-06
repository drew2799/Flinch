function Cl2Kl(Cl)
    C̃ = Cl./1_500
    Kl = norminvcdf.(0,1,C̃)
    return Kl
end
function Cl2Kl(Cl, comm; root=0)
    if MPI.Comm_rank(comm) == root
        C̃ = Cl./1_500
        Kl = norminvcdf.(0,1,C̃)
    else
        Kl = nothing
    end
    return Kl
end
function Kl2Cl(Kl, comm; root=0)
    if MPI.Comm_rank(comm) == root
        Cl = 1_500 .* normcdf.(0,1,Kl)
    else
        Cl = nothing
    end
    return Cl
end
@adjoint function Kl2Cl(Kl, comm; root=0)
    
    Cl = Kl2Cl(Kl, comm, root=root)
    
    function Kl2Cl_PB(adj_Cl)
        if MPI.Comm_rank(comm) == root
            adj_Kl = 1_500 .* pdf.(Normal(0,1), Kl) .* adj_Cl
        else
            adj_Kl = nothing
        end
        return (adj_Kl, nothing, nothing)
    end
    return Cl, Kl2Cl_PB
end

function DMapReparam(dmap, invN)
    sqrt_invN = DMap{RR, Float64}(sqrt.(invN.pixels), invN.info)
    reparam_dmap = sqrt_invN * dmap
    return reparam_dmap
end
@adjoint function DMapReparam(dmap, invN)
    rep_dmap = DMapReparam(dmap, invN)
    function DMapReparam_PB(adj_rep_dmap)
        adj_dmap = DMap{RR, Float64}(sqrt.(invN.pixels) .* adj_rep_dmap.pixels, invN.info) 
        return (adj_dmap, nothing)
    end
    return rep_dmap, DMapReparam_PB
end


function MapReparam(x, Σ⁻¹, comm; root=0)
    if MPI.Comm_rank(comm) == root
        y = HealpixMap{Float64, RingOrder}(sqrt.(Σ⁻¹)*x.pixels)
    else
        y = nothing
    end
    return y
end
@adjoint function MapReparam(x, Σ⁻¹, comm; root=0)
    y = MapReparam(x, Σ⁻¹, comm, root=root)
    function MapReparam_PB(ȳ)
        if MPI.Comm_rank(comm) == root
            x̄ = HealpixMap{Float64, RingOrder}(transpose(sqrt.(Σ⁻¹)) * ȳ)
        else
            x̄ = nothing
        end
        return (x̄, nothing, nothing, nothing)
    end
    return y, MapReparam_PB
end
