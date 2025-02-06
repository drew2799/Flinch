function nℓπ(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, BP_l=BP_l, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp = NegLogPosterior(θ, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, BP_l=BP_l, root = root)
    return nlp
end

function nℓπ_grad(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, BP_l=BP_l, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp_grad = gradient(x->NegLogPosterior(x, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, BP_l=BP_l, root=root), θ)
    return nlp_grad
end
