function Realization(realiz_Cl_file, nside, lmax, seed)

    Random.seed!(seed)

    realiz_Dl = CSV.read(realiz_Cl_file, DataFrame)[1:lmax-1,1]
    realiz_Cl = dl2cl(realiz_Dl, 2)
    realiz_Cl[1] += 1e-10
    realiz_Cl[2] += 1e-10

    realiz_HMap = synfast(realiz_Cl, nside)
    realiz_HAlm = map2alm(realiz_HMap, lmax=lmax)

    return realiz_Cl, realiz_HAlm, realiz_HMap
end

function Measurement(realiz_Cl, Bl, Pl, mask, noise, nside, lmax, seed)

    Random.seed!(seed)
    realiz_map = synfast(realiz_Cl.*(Bl.^2 .* Pl.^2), nside)

    e = rand(MvNormal(zeros(nside2npix(nside)), Diagonal(noise)))

    gen_HMap = HealpixMap{Float64,RingOrder}(deepcopy(realiz_map) + e)
    masked_gen_HMap = deepcopy(gen_HMap)
    masked_gen_HMap[mask.==1] .= 0

    gen_HAlm = map2alm(masked_gen_HMap, lmax=lmax)
    gen_Cl = anafast(masked_gen_HMap, lmax=lmax)

    return gen_Cl, gen_HAlm, masked_gen_HMap
end

function StartingPoint(gen_Cl, nside)

    start_HAlm = synalm(gen_Cl)
    start_Cl = alm2cl(start_HAlm)
    start_HMap = Healpix.alm2map(start_HAlm, nside)

    return start_Cl, start_HAlm, start_HMap
end

