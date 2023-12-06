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
                     bound::Bool = false,
                     weights::Union{Nothing, Vector{Real}} = nothing,
                     num_afc::Int = 1) where X <: Real
    
    if detections isa BitMatrix # Convert to vector of floats
        detections = vec(mean(detections, dims = 2))
    end
    if num_afc > 1 # Do we need to rescale
        detections = chance_rescale(detections, afc_chance(num_afc))
    end

    if (length(stimulus_levels) < 3 & bound == false)
        error("Minimum number of points for fitting is 3")
    end
    
    # Get init points for curve fitting
    dt_guess, kappa_guess = guess_sigmoid_coeffs(stimulus_levels, detections)

    if bound # Add extra top and tail
        x = [0; stimulus_levels; maximum(stimulus_levels) +  (maximum(stimulus_levels) - minimum(stimulus_levels)) * 0.5]
        y = [0; detections; 1]
    else
        x = stimulus_levels
        y = detections
    end

    if weights !== nothing
        if bound # Need to append the weights first
            weights = [maximum(weights)*5; weights; maximum(weights)*5]
        end
        if length(weights) !== length(x)
            error("Unequal number of stimulus levels and weight")
        end
        sig_fit = curve_fit(sigmoid, x, y, weights, [kappa_guess, dt_guess])
    else
        sig_fit = curve_fit(sigmoid, x, y, [kappa_guess, dt_guess])
    end
    # Make outputs
    dt_fit = sig_fit.param[2]
    jnd_fit = sigmoid2jnd(sig_fit.param)
    return sig_fit.param, dt_fit, jnd_fit
end

function guess_sigmoid_coeffs(x::Vector{X}, y::Vector{Float64}) where X <: Real
    # Nearest value to 0.5 is hopefully close to dt50
    dt_guess = x[findmin(abs.(y .- .5))[2]] # Nearest value to 0.5 is hopefully close to dt50
    # Then need to find first and last values near q1/q4
    q1_idx = findlast(y .<= 0.25)
    if q1_idx isa Nothing # No values below 0.25
        q1_idx = 1
    end
    q3_idx = findfirst(y .>= 0.75)
    if q3_idx isa Nothing # No values above 0.75
        q3_idx = length(x)
    end
    if q3_idx == q1_idx # Make sure they aren't the s
        q3_idx += 1
    end
    # Estimate JND and then convert to kappa
    kappa_guess = sigma2kappa(jnd2sigma((x[q3_idx] - x[q1_idx]) / 2))
    if kappa_guess == Inf
        error("something")
    end
    return dt_guess, kappa_guess
end

function discretize_responses(stimulus_levels::Vector{X},
                              responses::Union{BitVector, Vector{Float64}};
                              num::Int = 6) where X <: Real
    # First check if any values are NaNs
    if any(isnan.(stimulus_levels))
        filter_idx = .!isnan.(stimulus_levels)
        stimulus_levels = stimulus_levels[filter_idx]
        responses = responses[filter_idx]
    end
    # Create bin edges and centers
    bin_edges = LinRange(percentile(stimulus_levels, 5) * 0.95, percentile(stimulus_levels, 95) * 1.05, num)
    bin_means = bin_edges[1:end-1] .+ (bin_edges[2] - bin_edges[1]) / 2
    # Get bin index for each stimulus level
    weight, bin_idx = histcountindices(stimulus_levels, bin_edges)
    # Get the mean number of responses for each bin
    pd = [mean(responses[bin_idx .== x]) for x in range(1, num-1)];
    if any(isnan.(pd))
        filter_idx = .!isnan.(pd)
        pd = pd[filter_idx]
        bin_means = bin_means[filter_idx]
        weight = weight[filter_idx]
    end

    return collect(bin_means), pd, weight
end


"""
chance_MLE(X::Vector{Int}, Y::Vector{Bool}, Chance::Float64, Coeffs::Vector{Float64})

Custom loss function for logistic regression. Scales probabilities by chance to prevent over-fitting
    to false alarms. Returns the inverted likelihood (loss) such that minimization optimizers can be used.

returns loss
"""
function chance_MLE(X::Vector{Int64},
                    Y::Union{BitVector, Vector{Bool}},
                    Chance::Float64,
                    Coeffs::Vector{Float64})
    log_odds_projected = X .* Coeffs[1] .+ Coeffs[2] # Project in form y = mx + c
    # Convert to probability while adjusting for chance
    probabilities = (exp.(log_odds_projected) ./ (1 .+ exp.(log_odds_projected))) .* Chance .+ Chance
    # Compute likelihoods
    likelihood_detected = [log(x) for (x, y) in zip(probabilities, Y) if y];
    likelihood_not_detected = [log(1 - x) for (x, y) in zip(probabilities, Y) if !y];
    total_likelihood = sum(likelihood_detected) + sum(likelihood_not_detected)
    return total_likelihood * -1
end