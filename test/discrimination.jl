using DrWatson, Distributions, UnicodePlots, LsqFit, GLM, Printf
@quickactivate "PsychophysicsSimulations"
include(srcdir("psycho_sim_utils.jl"));

## Define the underlying psychometric function
    x = collect(0:2:100)
    mu = 50
    sigma = 10

    d = Normal(mu, sigma)
    pd = cdf.(d, x)
    lineplot(x, pd)

## Test samples at different intensities (assuming the same sigma)
    num_trials = 10
    amps = collect(30:10:70)
    std_draws = rand(d, num_trials, length(amps));
    comp_draws = fill(0.0, num_trials, length(amps));
    for (i, a) in enumerate(amps)
        comp_draws[:,i] = rand(Normal(a, sigma), num_trials)
    end
    # Get mean number of times the comparison stimulus was judged as higher
    comp_higher = vec(mean(comp_draws .> std_draws, dims = 1))
    # Plot overlay
    l = lineplot(x, pd);
    scatterplot!(l, amps, comp_higher);
    display(l)
    ~, ~, jdn = fit_sigmoid(amps, comp_higher, Bound = true)
    @printf("Predicted Sigma = %0.2f", jnd2sigma(jdn))

## Randomly selected amplitudes
    valid_amps = collect(0:2:100);
    num_trials = 100;
    stim_amps = rand(valid_amps, num_trials);
    std_draws = rand(d, num_trials);
    comp_draws = [rand(Normal(x, sigma)) for x in stim_amps];
    detections = comp_draws .> std_draws;
    scatterplot(stim_amps, detections)

## Amp
    valid_amps = collect(30:2:70);
    num_trials = 100;
    stim_amps = rand(valid_amps, num_trials);
    std_draws = vec(rand(Normal(mu, sigma), num_trials));
    comp_draws = [rand(Normal(x, sigma)) for x in stim_amps];
    comp_higher = comp_draws .> std_draws;
    scatterplot(stim_amps, comp_higher)
    # Deviation
    stim_greater_idx = stim_amps .> mu;
    deviation = abs.(stim_amps .- mu);
    discriminated = fill(0, num_trials);
    for (i, (s, dt)) in enumerate(zip(stim_amps, comp_higher))
        if ((s > mu) & dt) | ((s < mu) & !dt)
            discriminated[i] = 1
        end
    end
    # scatterplot(abs.(stim_amps .- mu), abs.(discriminated))

    Xe = LinRange(0, 30, 6)
    Xm = Xe[1:end-1] .+ (Xe[2] - Xe[1]) / 2
    (_, idx) = histcountindices(deviation, Xe)
    pd = [mean(discriminated[idx .== x]) for x in sort(unique(idx))]
    scatterplot(Xm, pd)

    # Can we apply quest to this?? Inflection point should be at 50%