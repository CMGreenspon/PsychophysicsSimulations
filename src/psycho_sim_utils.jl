using DrWatson, Distributions, DataFrames, GLM
@quickactivate "PsychophysicsSimulations"
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


"""
    fit_sigmoid(x::Vector{Number}, y::Vector{Bool}; method::Symbol = :lsq)
    Fit a sigmoid using maximum-likelihood-estimation
"""
function fit_sigmoid(x::Vector{X},
                     y::BitVector) where X <: Real
    # Initalize variables such that if fit fails the return does not error
    estimated_threshold = NaN
    estimated_kappa = NaN
    try
        # Logit function predicting detection from stimulus
        fm = @formula(detection ~ stimulus)
        temp_df = DataFrame(detection = y, stimulus = x) 
        logit = glm(fm, temp_df, Binomial(), LogitLink())
        fit_coeffs = coef(logit)
        # Convert and save coeffs
        estimated_threshold = abs(fit_coeffs[1])/fit_coeffs[2]
        estimated_kappa = fit_coeffs[2]
    catch e
        println(e)
    end
    return estimated_threshold, estimated_kappa
end


"""
    fit_sigmoid(x::Vector, y::Union{BitMatrix, Vector{Float64}}; Bound = false)
    Fit a sigmoid with least-squares regression when there are a equal number
    of observations for each stimulus level.
"""
function fit_sigmoid(stimulus_levels::Vector{X},
                     detections::Union{BitMatrix, Vector{Float64}};
                     Bound::Bool = false,
                     num_afc::Int = 1) where X <: Real
    
    if detections isa BitMatrix # Convert to vector of floats
        detections = vec(mean(detections, dims = 2))
    end
    if num_afc > 1 # Do we need to rescale
        detections = chance_rescale(detections, afc_chance(num_afc))
    end
    
    # Get init points for optimization
    dt_guess = stimulus_levels[findmin(abs.(detections .- .5))[2]] # Nearest value to 0.5 is hopefully close to dt50
    # Then need to find first and last values near q1/q4
    q1_idx = findlast(detections .<= 0.25)
    if q1_idx isa Nothing # No values below 0.25
        q1_idx = 1
    end
    q3_idx = findfirst(detections .>= 0.75)
    if q3_idx isa Nothing # No values above 0.75
        q3_idx = length(stimulus_levels)
    end
    jnd_guess = (stimulus_levels[q3_idx] - stimulus_levels[q1_idx]) / 2
    kappa_guess = sigma2kappa(jnd2sigma(jnd_guess))

    if Bound # Add extra top and tail
        x = [0; stimulus_levels; maximum(stimulus_levels) +  (maximum(stimulus_levels) - minimum(stimulus_levels)) * 0.5]
        y = [0; detections; 1]
        sig_fit = curve_fit(sigmoid, x, y, [kappa_guess, dt_guess])
    else
        sig_fit = curve_fit(sigmoid, stimulus_levels, detections, [kappa_guess, dt_guess])
    end
    # Make outputs
    dt_fit = sig_fit.param[2]
    jnd_fit = sigmoid2jnd(sig_fit.param)
    return sig_fit.param, dt_fit, jnd_fit
end
