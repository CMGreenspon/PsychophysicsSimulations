using DrWatson, Distributions
@quickactivate "PsychophysicsSimulations"
# Series of helper/utility functions for PsychophysicsSimulations
"""
    AFCChance(NumAFC::Int) -> chance::Float
    Helper function to return chance for a given alternate-forced-choice experiment.
"""
function AFCChance(NumAFC::Int)
    if NumAFC < 1
        error("NumAFC < 1")
    elseif NumAFC == 1
        chance = 0.0 # Float
    else
        chance = 1 / NumAFC
    end
    return chance
end

"""
    ChanceRescale(x::Vector{AbstractFloat}, chance::Float) -> x_rescaled::Vector{AbstractFloat}
    Scales detection probabilities based on chance detection rate
"""
function ChanceRescale(x::Vector{Float64}, chance::Float64)
    if chance < 1.0 # Do not evaluate on 1 AFC version
        x = (x .- chance) ./ (1 - chance) # Scale for chance
        x[x.<0] .= 0 # Remove values below 0
    end
    return x
end

"""
    jnd2sigma(j::Real) -> sigma::Float
    Convert measured jnd to sigma of normal distribution
"""
function jnd2sigma(jnd::Real) 
    sigma = (1 / quantile(Normal(), 0.75)) * jnd # Convert JND to sigma
    return sigma
end

"""
    sigma2kappa(sigma::Real) -> kappa::Float
    Estimate kappa of sigmoid function from sigma of normal distribution
"""
function sigma2kappa(sigma::Real)
    kappa =  1.7 / sigma; # Convert sigma to kappa
    return kappa
end

"""
    sigmoid(x::Vector, coeffs::Vector) -> y::Vector{Float}
    Sigmoid function as a 2-coefficient logistic function that takes mu and scale (kappa) terms
"""
function sigmoid(x::Vector, coeffs::Vector)
    if length(coeffs) != 2
        error("Coeffs must havea length of 2")
    end
    y = 1 ./ (1 .+ exp.(-coeffs[1] .* (x .- coeffs[2])))
    return y
end

function FitSigmoid(x::Vector, y::Vector)

end