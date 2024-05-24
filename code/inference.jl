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

#%% --- Definitions ---
# load the 33PD data
df = CSV.read("../data/experiments/data_exp33PD.csv", DataFrame; header=1, skipto=3)

# we simulate until 1500 cells
max_cells = 1500

# experiments have 141 timepts
total_pts = 141

# list of column names for a treatment
DMSO = [
    Symbol("02_DMSO alltime"),
    Symbol("06_DMSO30 min removed"),
    Symbol("10_DMSO 2h removal")
]

TMZ10 = [
    Symbol("03_10um alltime"),
    Symbol("07_10um 30 min removed"),
    Symbol("11_10um 2h removal")
]

TMZ500 = [
    Symbol("04_500um alltime"),
    Symbol("08_500um 30 min removal"),
    Symbol("12_500um 2h removal")
]

# define model step
@everywhere step_methylate_g1!(a, m) = begin
    Glioblas.replicate!(a, m)
    Glioblas.methylate!(a, m; when=["es"])
    Glioblas.arrest!(a, m)
    Glioblas.apoptosis!(a, m)
end

@everywhere function calc_ic(col_names)
    return trunc(Int, mean((df[1, col_names])))
end

@everywhere function simulate(params)
    mdata = [nagents]
    _, mdf = paramscan(
        params,
        Glioblas.init_model;
        agent_step! = step_methylate_g1!,
        n = Glioblas.terminate,
        mdata,
        parallel=true
    )
    return mdf
end

@everywhere function experiment(p)
    DMSO_t0_cells = calc_ic(DMSO)
    TMZ10_t0_cells = calc_ic(TMZ10)
    TMZ500_t0_cells = calc_ic(TMZ500)

    l1, l2, L, p_drug10, p_drug500 = p
    l1, l2, L = (x -> trunc(Int,x)).([l1,l2,L])

    DMSO_params = Dict(
        :t0_cells => DMSO_t0_cells,
        :max_cells => max_cells,
        :stages => ("g1"=>l1,"es"=>l2,"g2"=>L-l1-l2),
        :t0_sampler => Glioblas.powerdecay_sampler,
        :p_drug => 0.0,
        :p_mgmt => 0.0,
        :max_t => total_pts-1,
        :rng => (x -> Random.MersenneTwister(x)).(2:5)
    );

    DMSO_mdf = simulate(DMSO_params)

    TMZ10_params = DMSO_params
    TMZ10_params[:t0_cells] = TMZ10_t0_cells
    TMZ10_params[:p_drug] = p_drug10

    TMZ10_mdf = simulate(TMZ10_params)

    TMZ500_params = DMSO_params
    TMZ500_params[:t0_cells] = TMZ500_t0_cells
    TMZ500_params[:p_drug] = p_drug500

    TMZ500_mdf = simulate(TMZ500_params)
    return DMSO_mdf, TMZ10_mdf, TMZ500_mdf
end

@everywhere function mean_trajectory(mdf)
    return combine(groupby(mdf[:, [:step, :nagents]], :step), :nagents => mean)[:, :nagents_mean]
end

@everywhere function length_hack(est::Vector)
    l = length(est)
    l < total_pts ? nothing : return est
    return [est; fill(est[end], total_pts-l)]
end

@everywhere function lsq_distance(treatment, estimate)
    distance = 0.0
    truth = mean(eachcol(df[:, treatment]))
    # normalize to match with estimates
    truth = truth / truth[11]
    distance += sum(((estimate .- truth) ./ truth) .^ 2.0)
    return distance
end

# define an error function
@everywhere function objective(p)
    (p[4] >= 0.) ? nothing : return Inf
    (p[5] >= 0.) ? nothing : return Inf
    DMSO_mdf, TMZ10_mdf, TMZ500_mdf = experiment(p)

    DMSO_est, TMZ10_est, TMZ500_est = mean_trajectory.([DMSO_mdf, TMZ10_mdf, TMZ500_mdf])

    # hack for parameters with early terminating trajectories
    DMSO_est, TMZ10_est, TMZ500_est = length_hack.([DMSO_est, TMZ10_est, TMZ500_est])

    # normalizing leads to better optimization
    DMSO_est, TMZ10_est, TMZ500_est = (x -> x/x[11]).([DMSO_est, TMZ10_est, TMZ500_est])

    distance = 0.0
    distance += lsq_distance(DMSO, DMSO_est)
    distance += lsq_distance(TMZ10, TMZ10_est)
    distance += lsq_distance(TMZ500, TMZ500_est)

    return distance
end

function clean_mdf(mdf, name)
    return rename!(combine(groupby(mdf[:, [:step, :nagents]], :step), :nagents => mean), :nagents_mean => name)
end

function length_hack(est::DataFrame)
    l = nrow(est)
    l < total_pts ? nothing : return est
    tail = DataFrame(Dict(:step => l:total_pts-1, propertynames(est)[2] => fill(est[end, 2], total_pts-l)))
    return append!(est, tail)
end

#%% --- Optimization ---
# bounds and initial guess
# l1, l2, L, p_drug10, p_drug500
lower = [0.0, 0.0, 20.0, 0.0, 0.0]
upper = [50.0, 50.0, 100.0, 0.5, 1.0]
initial_p = [3.0, 35.0, 81.0, 0.03, 0.122]
# inner_optimizer = GradientDescent()

# optimize paramters with particle swarm
result = try
    optimize(
        objective,
        initial_p,
        NelderMead(), #ParticleSwarm(;lower,upper,n_particles=50),
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

#%% --- Export ---
# run the model with inferred parameter to get the trajectory
DMSO_mdf, TMZ10_mdf, TMZ500_mdf = experiment(result.minimizer)

DMSO_est = length_hack(clean_mdf(DMSO_mdf, :DMSO))
TMZ10_est = length_hack(clean_mdf(TMZ10_mdf, :TMZ10))
TMZ500_est = length_hack(clean_mdf(TMZ500_mdf, :TMZ500))

# save the trajectories and paramters
CSV.write("../data/inference/fitted_exp33PD.csv", innerjoin(DMSO_est, TMZ10_est, TMZ500_est; on = :step));
writedlm("../data/inference/parameters.csv", Optim.minimizer(result), ',');