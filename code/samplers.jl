"""
# samplers.jl
samplers for initializing a cell population
"""

export uniform_sampler, powerdecay_sampler


function uniform_sampler(;
	n_samples::Int64,
	period::Int64,
	rng::AbstractRNG
    )

    d = rand(rng, 1:period, n_samples)

    return d
end


function powerdecay_sampler(;
	n_samples::Int64,
	period::Int64,
	rng::AbstractRNG
    )

    d = convert(
        Vector{Float64},
    	rand(rng, DiscreteUniform(1, period), n_samples)
    )
    u = rand(rng, Uniform(0.,1.), n_samples)
    p = d[(u .<= ((2.0log(2.0)) .^ -(d ./ convert(Float64, period))))]
    lacking_n = n_samples - length(p)

    if lacking_n > 0
        extra = powerdecay_sampler(;
        	n_samples=lacking_n,
        	period,
        	rng)
        p = vcat(p,extra)
    end

    return convert(Vector{Int64}, p)
end
