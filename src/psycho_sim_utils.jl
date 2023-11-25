# Series of helper/utility functions for PsychophysicsSimulations
"""
    afc_chance(num_afc::Int) -> chance::Float
    Helper function to return chance for a given alternate-forced-choice experiment.
"""
function afc_chance(num_afc::Int)
    if num_afc < 1
        error("num_afc < 1")
    elseif num_afc == 1
        chance = 0.0 # Float
    else
        chance = 1 / num_afc
    end
    return chance
end


"""
chance_resacle(x::Vector{AbstractFloat}, chance::Float) -> x_rescaled::Vector{AbstractFloat}
    Scales detection probabilities based on chance detection rate
"""
function chance_resacle(x::Vector{Float64}, chance::Float64)
    if chance < 1.0 # Do not evaluate on 1 AFC version
        x = (x .- chance) ./ (1 - chance) # Scale for chance
        x[x .< 0] .= 0 # Remove values below 0
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
    sigma2jnd(sigma::Real)
    Convert sigma of normal distribution to JND in stimulus units
"""
function sigma2jnd(sigma::Real) 
    jnd = sigma / (1 / quantile(Normal(), 0.75))
    return jnd
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
function sigmoid(x::Vector{X},
                 coeffs::Vector{Float64}) where X <: Real
    if length(coeffs) != 2
        error("Coeffs must havea length of 2")
    end
    y = 1 ./ (1 .+ exp.(-coeffs[1] .* (x .- coeffs[2])))
    return y
end


"""
    inverse_sigmoid(coeffs::Vector{Float64}, yq::Real)
    Inverse 2-coefficient sigmoid function return probability of y at x
"""
function inverse_sigmoid(coeffs::Vector{Float64}, yq::Real)
    if length(coeffs) != 2
        error("Coeffs must have a length of 2")
    end
    if 0 <= yq <= 1 == false
        error("yq must be a real number between 0 and 1")
    end

    y = (log(1/yq - 1) / -coeffs[1]) + coeffs[2]
    return y
end


function sigmoid2jnd(coeffs::Vector{Float64})
    jnd = (inverse_sigmoid(coeffs, 0.75) - inverse_sigmoid(coeffs, 0.25)) / 2
    return jnd
end
