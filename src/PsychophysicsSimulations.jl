module PsychophysicsSimulations
    using Reexport
    @reexport using Distributions, DataFrames, GLM, LsqFit, Optim, NaNStatistics, Printf, Random, StatsBase
    include("psycho_sim_utils.jl")
    include("fit_psychometric.jl")

end