using DrWatson
@quickactivate "PsychophysicsSimulations"

function ConstantStimulation(Stims::Vector, pDetected::Vector, NumReps::Int;
    NumPermutations::Int = 1, FitMethod::Symbol=:sigmoid, NumAFC::Int=2, BoundSig::Bool=true)
    # Compute chance
    chance = afc_chance(NumAFC)
    # Create once for comparison
    pD_Repeated = repeat(pDetected, inner=(1, NumReps))

    # Initialize outputs
    t_est = fill(NaN, NumPermutations)
    jnd_est = fill(NaN, NumPermutations)

    for p = 1:NumPermutations
        # Get the proportion of trials where the draw is below the p(detected) at each intensity
        pd = mean((rand(length(Stims), NumReps) .< pD_Repeated) .| # First draw greater than pd
                  (rand(length(Stims), NumReps) .< chance), dims=2)
        pd = ChanceRescale(pd, chance)
        # Get fair estimates of the detection threshold and jnd
        dt_idx = findmin(abs.(pd .- 0.5))[2]
        jnd_idx = [findmin(abs.(pd .- 0.25))[2], findmin(abs.(pd .- 0.75))[2]]
        jnd_est = (Stims[jnd_idx[2]] - Stims[jnd_idx[1]]) / 2
        k_est = sigma2k(jnd2sigma(jnd_est))
        # Fit a sigmoid to the values
        try
            if BoundSig
                sig_fit = curve_fit(sigmoid, [0; Stims; 100], vec([0; pd; 1]), [k_est, Stims[dt_idx]])
            else
                sig_fit = curve_fit(sigmoid, Stims, vec(pd), [k_est, Stims[dt_idx]])
            end
            if sig_fit.param[2] > 0 && sig_fit.param[2] < 100
                t_est[p] = sig_fit.param[2]
            end
        catch
            continue
        end
    end

    return t_est, jnd_est
end