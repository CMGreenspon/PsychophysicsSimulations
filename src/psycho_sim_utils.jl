using DrWatson, Distributions, DataFrames, GLM
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
function sigmoid(x::Vector{X}, coeffs::Vector{Float64}) where X <: Real
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
        error("Coeffs must havea length of 2")
    end
    y = (log(1/yq - 1) / -coeffs[1]) + coeffs[2]
    return y
end


"""
    FitSigmoid(x::Vector{Number}, y::Vector{Bool}; method::Symbol = :lsq)
    Fit a sigmoid using maximum-likelihood-estimation
"""
function FitSigmoid(x::Vector{X}, y::BitVector) where X <: Real
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
    FitSigmoid(x::Vector, y::BitMatrix; Bound = false)
    Fit a sigmoid with least-squares regression when there are a equal number
    of observations for each stimulus level.
"""
function FitSigmoid(
    stimulus_levels::Vector{X},
    detections::Union{BitMatrix, Vector{Float64}};
    Bound::Bool = false,
    NumAFC::Int = 1) where X <: Real
    
    if detections isa BitMatrix # Convert to vector of floats
        detections = vec(mean(detections, dims = 2))
    end
    if NumAFC > 1 # Do we need to rescale
        detections = ChanceRescale(detections, AFCChance(NumAFC))
    end
    
    # Get init points for optimization
    dt_guess = stimulus_levels[findmin(abs.(detections .- .5))[2]] # Nearest value to 0.5 is hopefully close to dt50
    # Then need to find first and last values near q1/q4
    q1_idx = findlast(detections .<= 0.25)
    if q1_idx isa Nothing
        q1_idx = 1
    end
    q3_idx = findfirst(detections .>= 0.75)
    if q3_idx isa Nothing
        q3_idx = length(valid_stims)
    end
    jnd_guess = (valid_stims[q3_idx] - valid_stims[q1_idx]) / 2
    kappa_guess = sigma2kappa(jnd2sigma(jnd_guess))

    if Bound # Add extra top and tail
        x = [0; valid_stims; maximum(valid_stims) +  (maximum(valid_stims) - minimum(valid_stims)) * 0.5]
        y = [0; detections; 1]
        sig_fit = curve_fit(sigmoid, x, y, [kappa_guess, dt_guess])
    else
        sig_fit = curve_fit(sigmoid, valid_stims, detections, [kappa_guess, dt_guess])
    end
    # Make outputs
    dt_fit = sig_fit.param[2]
    jnd_fit = (inverse_sigmoid(sig_fit.param, 0.75) - inverse_sigmoid(sig_fit.param, 0.25)) / 2
    return dt_fit, jnd_fit
end
