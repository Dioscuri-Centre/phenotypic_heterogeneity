using DelimitedFiles
using CSV
using DataFrames
using Random
Random.seed!(123);
using Optim
using Statistics

using Distributed
addprocs(4)
@everywhere begin
    using Pkg
    Pkg.activate("../")
    Pkg.instantiate()
    using Random
    using Agents
    include("Glioblas.jl")
end

# load the 33PD data
df = CSV.read("../data/experiments/data_exp33PD.csv", DataFrame; header=1, skipto=3)

# define model step
@everywhere step_methylate_g1!(a, m) = begin
    Glioblas.replicate!(a, m)
    Glioblas.methylate!(a, m; when=["es"])
    Glioblas.arrest!(a, m)
    Glioblas.apoptosis!(a, m)
end

# define an error function
@everywhere function lsq_distance(u)
    DMSO_t0_cells = trunc(Int, (df[1, Symbol("02_DMSO alltime")] + df[1, Symbol("06_DMSO30 min removed")] + df[1, Symbol("10_DMSO 2h removal")])/3)
    TMZ_t0_cells = trunc(Int, (df[1, Symbol("04_500um alltime")] + df[1, Symbol("08_500um 30 min removal")] + df[1, Symbol("12_500um 2h removal")])/3)

    l1, l2, L, growth_rate, drug_dose, death_rate = u
    l1 = trunc(Int,l1)
    l2 = trunc(Int,l2)
    L = trunc(Int,L)

    mdata = [nagents]

    DMSO = Dict(
        :t0_cells => DMSO_t0_cells,
        :max_cells => 1500,
        :stages => ("g1"=>l1,"es"=>l2,"g2"=>L-l1-l2),
        :t0_sampler => Glioblas.powerdecay_sampler,
        :growth_rate => growth_rate,
        :drug_dose => 0.0,
        :death_rate => death_rate,
        :mgmt_conc => 0.0,
        :max_t => 140,
        :rng => (x -> Random.MersenneTwister(x)).(1:4)
    );

    _, DMSO_mdf = paramscan(
        DMSO,
        Glioblas.init_model;
        agent_step! = step_methylate_g1!,
        n = Glioblas.terminate,
        mdata,
        parallel=true
    )

    TMZ = Dict(
        :t0_cells => TMZ_t0_cells,
        :max_cells => 1500,
        :stages => ("g1"=>l1,"es"=>l2,"g2"=>L-l1-l2),
        :t0_sampler => Glioblas.powerdecay_sampler,
        :growth_rate => growth_rate,
        :drug_dose => drug_dose,
        :death_rate => death_rate,
        :mgmt_conc => 0.0,
        :max_t => 140,
        :rng => (x -> Random.MersenneTwister(x)).(1:4)
    );

    _, TMZ_mdf = paramscan(
        TMZ,
        Glioblas.init_model;
        agent_step! = step_methylate_g1!,
        n = Glioblas.terminate,
        mdata,
        parallel=true
    )

    DMSO_est = combine(groupby(DMSO_mdf[:, [:step, :nagents]], :step), :nagents => mean)[:, :nagents_mean]
    TMZ_est = combine(groupby(TMZ_mdf[:, [:step, :nagents]], :step), :nagents => mean)[:, :nagents_mean]

    # hack for parameters with early terminating trajectories
    # this will not matter at the end, there will be abrupt jump to max_cell
    # and these parameters will not be selected
    DMSO_l = length(DMSO_est)
    TMZ_l = length(TMZ_est)
    DMSO_est = [DMSO_est; fill(DMSO[:max_cells],141-DMSO_l)]
    TMZ_est = [TMZ_est; fill(TMZ[:max_cells],141-TMZ_l)]

    # normalizing leads to better optimization
    DMSO_est = DMSO_est / DMSO_est[11]
    TMZ_est = TMZ_est / TMZ_est[11]

    distance = 0.0
    for target in ["02_DMSO alltime", "06_DMSO30 min removed", "10_DMSO 2h removal"]
        # normalize data to match estimated trajectories
        DMSO = df[:, Symbol(target)] / df[11, Symbol(target)]
        distance += sum(((DMSO_est .- DMSO) ./ DMSO) .^ 2.0)
    end

    for target in ["04_500um alltime", "08_500um 30 min removal", "12_500um 2h removal"]
        # normalize data to match estimated trajectories
        TMZ = df[:, Symbol(target)] / df[11, Symbol(target)]
        distance += sum(((TMZ_est .- TMZ) ./ TMZ) .^ 2.0)
    end

    return distance
end

# bounds and initial guess
lower = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
upper = [40.0, 40.0, 100.0, 1.0, 1.0, 1.0]
initial_x = [1.0, 40.0, 80.0, 1.0, 0.115, 1.0]
# inner_optimizer = GradientDescent()

# throw away run for compilation
optimize(
    lsq_distance,
    initial_x,
    ParticleSwarm(;lower,upper,n_particles=3),
    Optim.Options(
        show_trace = false,
        show_every = 1,
        time_limit = 2
    )
)

# optimize paramters with particle swarm
result = try
    optimize(
        lsq_distance,
        initial_x,
        ParticleSwarm(;lower,upper,n_particles=50),
        Optim.Options(
            show_trace = true,
            show_every = 25,
            time_limit = 120
        )
    )
catch e
    println(e)
finally
    rmprocs(workers())
end

# run the model with inferred parameter to get the trajectory
DMSO_t0_cells = trunc(Int, (df[1, Symbol("02_DMSO alltime")] + df[1, Symbol("06_DMSO30 min removed")] + df[1, Symbol("10_DMSO 2h removal")])/3)
TMZ_t0_cells = trunc(Int, (df[1, Symbol("04_500um alltime")] + df[1, Symbol("08_500um 30 min removal")] + df[1, Symbol("12_500um 2h removal")])/3)

l1, l2, L, growth_rate, drug_dose, death_rate = Optim.minimizer(result)
l1 = trunc(Int,l1)
l2 = trunc(Int,l2)
L = trunc(Int,L)

mdata = [nagents]

DMSO = Dict(
    :t0_cells => DMSO_t0_cells,
    :max_cells => 1500,
    :stages => ("g1"=>l1,"es"=>l2,"g2"=>L-l1-l2),
    :t0_sampler => Glioblas.powerdecay_sampler,
    :growth_rate => growth_rate,
    :drug_dose => 0.0,
    :death_rate => death_rate,
    :mgmt_conc => 0.0,
    :max_t => 140,
    :rng => (x -> Random.MersenneTwister(x)).(1:4)
);

_, DMSO_mdf = paramscan(
    DMSO,
    Glioblas.init_model;
    agent_step! = step_methylate_g1!,
    n = Glioblas.terminate,
    mdata,
    parallel=true
)

TMZ = Dict(
    :t0_cells => TMZ_t0_cells,
    :max_cells => 1500,
    :stages => ("g1"=>l1,"es"=>l2,"g2"=>L-l1-l2),
    :t0_sampler => Glioblas.powerdecay_sampler,
    :growth_rate => growth_rate,
    :drug_dose => drug_dose,
    :death_rate => death_rate,
    :mgmt_conc => 0.0,
    :max_t => 140,
    :rng => (x -> Random.MersenneTwister(x)).(1:4)
);

_, TMZ_mdf = paramscan(
    TMZ,
    Glioblas.init_model;
    agent_step! = step_methylate_g1!,
    n = Glioblas.terminate,
    mdata,
    parallel=true
)

DMSO_est = rename!(combine(groupby(DMSO_mdf[:, [:step, :nagents]], :step), :nagents => mean), :nagents_mean => :DMSO)
TMZ_est = rename!(combine(groupby(TMZ_mdf[:, [:step, :nagents]], :step), :nagents => mean), :nagents_mean => Symbol("500um"))

# save the trajectories and paramters
CSV.write("../data/inference/fitted_exp33PD.csv", innerjoin(DMSO_est, TMZ_est, on = :step));
writedlm("../data/inference/parameters.csv", Optim.minimizer(result), ',');