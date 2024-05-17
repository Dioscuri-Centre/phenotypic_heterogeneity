"""
# measures.jl
samplers for initializing a cell population
"""

export infer_stage, counter_freq, stage_freq


function infer_stage(;
    counter::Int64,
    stages::NTuple{T, Pair{String, Int64}} where T
)

    @assert counter <= sum((x->x.second).(stages))

    # start with first stage index
    s = 1;
    stage = Tuple((x->x.first).(stages))[s]

    # if the counter is larger
    # than the sum of k stages
    # update the stage
    while counter > sum((x->x.second).(stages)[1:s])
        s+=1
        stage = Tuple((x->x.first).(stages))[s]
    end

    return stage
end


function counter_freq(model::ABM)
    counters = [agent.counter for agent in values(model.agents)]
    return [count(==(counter), counters) for counter in 1:model.period]
end


function stage_freq(model::ABM)
    slist = [agent.stage for agent in values(model.agents)]
    return [count(==(stage), slist) for stage in (x->x.first).(model.stages)]
end
