using DrWatson, Distributions, UnicodePlots
@quickactivate "PsychophysicsSimulations"
include(srcdir("psycho_sim_utils.jl"))

## Create pdf of detection probability
    valid_stims = collect(10:10:100) # These are the amplitudes that can be given 
    detection_threshold = 50 # microamps
    jnd = 10 # microamps
    sigma = jnd2sigma(jnd)
    psychometric_pdf = Normal(detection_threshold, sigma)
    pDetected = cdf(psychometric_pdf, valid_stims)

    lineplot(valid_stims, pDetected)

## Run a single simulation
function DT1(valid_stims, pDetected; NumPerms = 1000)
    dt = fill(0.0, length(valid_stims), NumPerms)
    for p = 1:NumPerms
        chance = 0.5
        NumReps = 10
        if chance < 1.0
            detections = mean(mapslices(x -> x .< pDetected , rand(length(valid_stims), NumReps), dims = 1) .| 
                            mapslices(x -> x .< chance , rand(length(valid_stims), NumReps), dims = 1), dims = 2)
            detections = ChanceRescale(vec(detections), chance)
        else
            detections = mean(mapslices(x -> x .< pDetected , rand(length(valid_stims), NumReps), dims = 1), dims = 2)
        end
        dt[:,p] = detections
    end
end

function DT2(valid_stims, pDetected; NumPerms = 1000)
    NumReps = 10
    chance = 0.5
    pD_Repeated = repeat(pDetected, inner=(1, NumReps))
    dt = fill(0.0, length(valid_stims), NumPerms)
    for p = 1:NumPerms
        detections = mean(
                    (rand(length(valid_stims), NumReps) .< pD_Repeated) .| # First draw greater than pd
                    (rand(length(valid_stims), NumReps) .< chance), dims=2)
        detections = ChanceRescale(vec(detections), chance)
        dt[:,p] = detections
    end
end