"""
# models.jl
Agent-based model(s) and related entities
"""

export Cell, init_model
export concat_step!, replicate!, methylate!, demethylate!, apoptosis!, arrest!
export terminate, terminate_arrested

# initialize a NoSpaceAgent called Cell
@agent Cell NoSpaceAgent begin
    counter::Int64
    stage::String
    methylated::Bool
    arrested::Bool
end


# function to initialize agent-based model
function init_model(;
    t0_cells::Int64,
    max_cells::Int64,
    stages::NTuple{T, Pair{String,Int64}} where T,
    t0_sampler::Union{Vector{Int64}, Function},
    growth_rate::Float64,
    drug_dose::Float64, # Tm+ methylation rate
    death_rate::Float64, # d drug dose
    mgmt_conc::Float64, # Tm- demethylration rate
    max_t::Int64,
    rng::AbstractRNG
)

    # some sanity checks
    @assert t0_cells > 0
    @assert t0_cells <= max_cells

    # length of cell cycle
    period = sum((x->x.second).(stages))

    @assert period .>= 0

    # @assert growth_rate >= 0.0
    # @assert growth_rate / period <= 1.0

    # @assert drug_dose / period >= 0.0
    # @assert drug_dose / period <= 1.0

    # @assert death_rate / period >= 0.0
    # @assert death_rate / period <= 1.0

    # @assert mgmt_conc / period >= 0.0
    # @assert mgmt_conc / period <= 1.0

    space = nothing
    properties = (;
        t0_cells,
        max_cells,
        stages,
        period,
        growth_rate,
        drug_dose,
        death_rate,
        mgmt_conc,
        max_t,
        rng
    )

    scheduler = Schedulers.randomly

    # initialize model
    model = ABM(Cell, space; properties, rng, scheduler)

    # this is the initial counter frequency
    # of cells over cell cycle counter
    if t0_sampler isa Vector{Int64}
        t0_counters = t0_sampler
    else
        t0_counters = t0_sampler(; n_samples=t0_cells, period, rng)
    end

    for counter in t0_counters
        # infer the stage fron the counter
        stage = infer_stage(; counter, stages)
        # add an agent
        add_agent!(model, counter, stage, false, false) # methylated false, arrested false
    end

    return model
end


function replicate!(agent, model)

    n = nagents(model)
    max_cells = model.max_cells
    stages = model.stages
    period = model.period
    rng = model.rng
    growth_rate = model.growth_rate

    stage_1 = first((x->x.first).(stages))

    # replicate
    if (agent.counter >= period) && (n < max_cells) && (agent.arrested == false)
        if rand(rng, Bernoulli(growth_rate))
            # reset counter of the current agent
            agent.counter = 1
            agent.stage = stage_1

            # add a new agent with counter 1
            add_agent!(model, 1, stage_1, false, false)
        end
    end

    # increase counter
    if (agent.counter < period) && (agent.arrested == false)
        agent.counter += 1
        # update stage
        agent.stage = infer_stage(; counter=agent.counter, stages)
    end

    return nothing
end


function methylate!(agent, model; when=["g1","es","g2"])

    period = model.period
    drug_dose = model.drug_dose
    rng = model.rng

    if (agent.methylated == false) && (agent.stage in when)
        agent.methylated = rand(rng, Bernoulli(drug_dose))
    end

    return nothing
end


function demethylate!(agent, model; when=["g1","es","g2"])

    period = model.period
    mgmt_conc = model.mgmt_conc
    rng = model.rng

    # mgmt demethylates cell
    if (agent.methylated == true) && (agent.stage in when)
        if rand(rng, Bernoulli(mgmt_conc))
            agent.methylated = false
        end
    end

    return nothing
end


function apoptosis!(agent, model; when=["g2"])

    period = model.period
    death_rate = model.death_rate
    rng = model.rng

    # trigger apoptosis for methylated cells in G2
    if (agent.methylated == true) && (agent.arrested == false) && (agent.stage in when)
        if rand(rng, Bernoulli(death_rate))
            remove_agent!(agent, model)
        end
    end

    return nothing
end


function arrest!(agent, model; when=["g2"])

    period = model.period
    arrest_rate = model.death_rate
    rng = model.rng

    # cell cycle arrest if methylated and in G2
    if (agent.methylated == true) && (agent.arrested == false) && (agent.stage in when)
        agent.arrested = rand(rng, Bernoulli(arrest_rate))
    end

    return nothing
end


function terminate(model, s)
    return (nagents(model) == 0) ||
    (sum(map(x -> x.arrested, values(model.agents))) == nagents(model)) ||
    (nagents(model) >= model.max_cells) ||
    (s >= model.max_t)
end
