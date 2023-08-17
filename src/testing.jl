using DrWatson, Distributions, UnicodePlots, LsqFit, GLM
@quickactivate "PsychophysicsSimulations"
include(srcdir("psycho_sim_utils.jl"))

## Create pdf of detection probability
    valid_stims = collect(10:10:100) # These are the amplitudes that can be given 
    detection_threshold = 50 # microamps
    jnd = 20 # microamps
    sigma = jnd2sigma(jnd)
    psychometric_pdf = Normal(detection_threshold, sigma)
    pDetect = cdf(psychometric_pdf, valid_stims)

    lineplot(valid_stims, pDetect)

## Fitting options
    BoundSig = false
    NumReps = 10
    NumAFC = 2
    chance = AFCChance(NumAFC)
    # Create once for comparison
    pD_Repeated = repeat(pDetect, inner=(1, NumReps))
    detections = (rand(length(valid_stims), NumReps) .< pD_Repeated) .| # First draw greater than pd
                 (rand(length(valid_stims), NumReps) .< chance)
    
    # Least squares

    # Fit a sigmoid to the values
    method = :lsq
    if method == :lsq
        # Compute detection probability adjusted for chance
        pDetected = ChanceRescale(vec(mean(detections, dims = 2)), chance)
        # Get init points for optimization
        dt_guess = valid_stims[findmin(abs.(pDetected .- .5))[2]] # Nearest value to 0.5 is hopefully close to dt50
        # Then need to find first and last values near q1/q4
        q1_idx = findlast(pDetected .<= 0.25)
        if q1_idx isa Nothing
            q1_idx = 1
        end
        q3_idx = findfirst(pDetected .>= 0.75)
        if q3_idx isa Nothing
            q3_idx = length(valid_stims)
        end
        jnd_guess = (valid_stims[q3_idx] - valid_stims[q1_idx]) / 2
        kappa_guess = sigma2kappa(jnd2sigma(jnd_guess))

        if BoundSig # Add extra top and tail
            x = [0; valid_stims; maximum(valid_stims) +  (maximum(valid_stims) - minimum(valid_stims)) * 0.5]
            y = [0; pDetected; 1]
            sig_fit = curve_fit(sigmoid, x, y, [kappa_guess, dt_guess])
        else
            sig_fit = curve_fit(sigmoid, valid_stims, pDetected, [kappa_guess, dt_guess])
        end
        # Make outputs
        dt_fit = sig_fit.param[2]
        jnd_fit = (inverse_sigmoid(sig_fit.param, 0.75) - inverse_sigmoid(sig_fit.param, 0.25)) / 2
    elseif method == :mle
        x = repeat(valid_stims, NumReps)
        y = collect(Iterators.flatten(detections))

        
    end

    println(
        """
        DT50 = $(round(dt_fit, digits = 2)) μA,
        JND = $(round(jnd_fit, digits = 2)) μA
        """
    )