using DrWatson, UnicodePlots, Revise, Base.Threads
@quickactivate :PsychophysicsSimulations

## Define the underlying psychometric function
    x = collect(0:2:100)
    mu = 50
    sigma = 10

    d = Normal(mu, sigma)
    pd = cdf.(d, x)
    lineplot(x, pd)

## Can we apply quest to this?? Inflection point should be at 50%
    min_amp = 0
    max_amp = 100
    num_trials = 50
    init_ratio = 0.1
    init_trials = 3
    num_perms = 1000
    dt_guess = fill(NaN, num_perms, 3);
    jnd_guess = fill(NaN, num_perms, 3);
    num_errors = 0
    for p = 1:num_perms
        amp = fill(0, num_trials);
        disc = falses(num_trials);

        # Predefine first N trials - we'll do this better in the future
        init_amps = shuffle(repeat([mu - mu * init_ratio,
                                    mu + mu * init_ratio,
                                    mu - mu * init_ratio * 2,
                                    mu + mu * init_ratio * 2], init_trials))
        init_amps = convert.(Int64, round.(init_amps ./ 2) .* 2)
        num_init_amps = init_trials * 4

        for t = 1:num_trials
            # Select the stim amp for the trial
            if t <= num_init_amps
                stim_amp = init_amps[t]
            else
                # Convert disc to abs deviation
                et, ek = PsychophysicsSimulations.fit_sigmoid(abs.(amp[1:t-1] .- mu), disc[1:t-1])
                # Select the next amplitude to balance around standard
                if mean(amp[1:t]) > mu
                    stim_amp = mu - abs(et)
                else
                    stim_amp = mu + abs(et)
                end
                if isnan(stim_amp)
                    stim_amp = rand(init_amps)
                end
            end
            # Validate stim_amp
            try
                stim_amp = convert(Int64, trunc(round(stim_amp / 2) * 2))
            catch
                stim_amp = rand(init_amps)
            end

            if stim_amp > max_amp
                stim_amp = max_amp
            elseif stim_amp < min_amp
                stim_amp = min_amp
            end
            amp[t] = stim_amp

            # Discriminate
            std_draw = rand(Normal(mu, sigma))
            comp_draw = rand(Normal(amp[t], sigma))
            if (comp_draw > std_draw) & (amp[t] > mu) | (comp_draw < std_draw) & (amp[t] < mu)
                disc[t] = true
            end
        end
        # lineplot(1:num_trials, amp)
        # try
            bin_means, pd, weight = PsychophysicsSimulations.discretize_responses(abs.(amp .- mu), disc)
            _, dt_guess[p, 1], jnd_guess[p, 1] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd, bound = true)
            _, dt_guess[p, 2], jnd_guess[p, 2] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd, counts = weight, bound = true)
            # _, dt_guess[p, 3], jnd_guess[p, 3] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd, counts = sqrt.(weight))
        # catch e
        #     println(e)
        #     println(bin_means)
        #     println(pd)
        #     println(weight)
        # end
    end
return
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
    ~, ~, jdn = PsychophysicsSimulations.fit_sigmoid(amps, comp_higher, bound = true)
    @printf("Predicted Sigma = %0.2f\n", PsychophysicsSimulations.jnd2sigma(jdn))

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

## Can we apply quest to this?? Inflection point should be at 50%
    min_amp = 0
    max_amp = 100
    num_trials = 50
    init_ratio = 0.1
    init_trials = 3
    num_perms = 1000
    dt_guess = fill(NaN, num_trials, 3)
    jnd_guess = fill(NaN, num_trials, 3)
    p = 1
    # @threads for p = 1:num_perms
        amp = fill(0, num_trials)
        disc = falses(num_trials)

        # Predefine first N trials - we'll do this better in the future
        init_amps = shuffle(repeat([mu - mu * init_ratio,
                                    mu + mu * init_ratio,
                                    mu - mu * init_ratio * 2,
                                    mu + mu * init_ratio * 2], init_trials))
        init_amps = convert.(Int64, round.(init_amps ./ 2) .* 2)
        num_init_amps = init_trials * 4

        for t = 1:num_trials
            # Select the stim amp for the trial
            if t <= num_init_amps
                stim_amp = init_amps[t]
            else
                # Convert disc to abs deviation
                et, ek = PsychophysicsSimulations.fit_sigmoid(abs.(amp[1:t-1] .- mu), disc[1:t-1])
                # Select the next amplitude to balance around standard
                if mean(amp[1:t]) > mu
                    stim_amp = mu - abs(et)
                else
                    stim_amp = mu + abs(et)
                end
            end

            # Validate stim_amp
            stim_amp = convert(Int64, trunc(round(stim_amp / 2) * 2))
            if stim_amp > max_amp
                stim_amp = max_amp
            elseif stim_amp < min_amp
                stim_amp = min_amp
            end
            amp[t] = stim_amp

            # Discriminate
            std_draw = rand(Normal(mu, sigma))
            comp_draw = rand(Normal(amp[t], sigma))
            if (comp_draw > std_draw) & (amp[t] > mu) | (comp_draw < std_draw) & (amp[t] < mu)
                disc[t] = true
            end
        end
        # try
        bin_means, pd, weight = PsychophysicsSimulations.discretize_responses(amp, disc)
        _, dt_guess[p, 1], jnd_guess[p, 1] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd)
        _, dt_guess[p, 2], jnd_guess[p, 2] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd, counts = weight)
        # _, dt_guess[p, 3], jnd_guess[p, 3] = PsychophysicsSimulations.fit_sigmoid(bin_means, pd, counts = sqrt.(weight))
    #     catch e
    #         println(e)
    #         println(bin_means)
    #         println(pd)
    #         println(weight)
    #     end
    # end