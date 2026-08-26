#!/usr/bin/env julia

"""
Restartable exact-model attractor and multistability pipeline.

Validated P1 states from the completed exact map and stable branch-switched
P2 checkpoints are opened read-only and used only as initial states. Every
output is written below `multistability_nonregularized_Dx4`.

Shell example

    julia --project=. --threads=16 src/nonregularized_multistability/scripts/run_nonregularized_multistability_pipeline.jl --run-id=exact_v1 --guide-run-id=canonical_v1 --guide-eta=0.001 --guide-source-profile=final --guide-atlas-profile=final --profile=final

REPL example on Linux or Windows

    include("src/nonregularized_multistability/scripts/run_nonregularized_multistability_pipeline.jl")
    result = run_sax_nonregularized_multistability_unattended()
"""

if !isdefined(@__MODULE__, :run_sax_regularized_bifurcation_pipeline)
    include(joinpath(
        @__DIR__, "..", "..", "regularized_bifurcation", "scripts",
        "run_regularized_bifurcation_pipeline.jl"))
end
if !isdefined(@__MODULE__, :SaxNonregularizedMultistabilitySettings)
    include(joinpath(
        @__DIR__, "..", "analysis",
        "sax_nonregularized_multistability_atlas.jl"))
end

if !isdefined(@__MODULE__, :SaxNonregularizedMultistabilityPipelineOptions)
    Base.@kwdef struct SaxNonregularizedMultistabilityPipelineOptions
        fingering::String = "Dx4"
        run_id::String = SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID
        profile::Symbol = :pilot
        guide_run_id::String = SAX_REGULARIZED_CANONICAL_RUN_ID
        guide_eta::Float64 = 1e-3
        guide_source_profile::Symbol = :final
        guide_atlas_profile::Symbol = :final
        stages::Tuple{Vararg{Symbol}} = (
            :guided, :propagate, :dh_repair,
            :validate, :edge, :assemble)
        resume::Bool = true
        status_only::Bool = false
        parallel_points::Bool = true
        parallel_validation::Bool = true
        gamma_range::Union{Nothing,Tuple{Float64,Float64}} = nothing
        gamma_points::Union{Nothing,Int} = nothing
        zeta_range::Union{Nothing,Tuple{Float64,Float64}} = nothing
        zeta_step::Union{Nothing,Float64} = nothing
        verbosity::Int = 1
    end
end

function _sax_nonregularized_multistability_help()
    return """
    Usage:
      julia --project=. --threads=16 src/nonregularized_multistability/scripts/run_nonregularized_multistability_pipeline.jl [options]

    Options:
      --fingering=Dx4
      --run-id=exact_v1
      --profile=smoke|pilot|final
      --guide-run-id=canonical_v1
      --guide-eta=0.001
      --guide-source-profile=final
      --guide-atlas-profile=final
      --only=guided,propagate,dh_repair,validate,edge,assemble
      --gamma-range=LOW,HIGH
      --gamma-points=N
      --zeta-range=LOW,HIGH
      --zeta-step=VALUE
      --serial-points
      --serial-validation
      --parallel-validation
      --status-only
      --no-resume
      --verbosity=N
      -h, --help

    Exact return-map validations use the Julia threads by default. Dense 40x4
    Periodic-Schur audits are sparse and internally serialized, so only one
    collocation factorization is resident at a time. Final also requires the
    compatible canonical stable-P2 checkpoint set.
    """
end

function _sax_nonregularized_parse_range(value::AbstractString, name::String)
    fields = split(value, ",")
    length(fields) == 2 || throw(ArgumentError(
        "$(name) must contain LOW,HIGH"))
    selected = (parse(Float64, fields[1]), parse(Float64, fields[2]))
    selected[1] <= selected[2] || throw(ArgumentError(
        "$(name) must be ordered"))
    return selected
end

function _parse_sax_nonregularized_multistability_args(args)
    values = Dict{Symbol,Any}(
        :fingering => "Dx4",
        :run_id => SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        :profile => :pilot,
        :guide_run_id => SAX_REGULARIZED_CANONICAL_RUN_ID,
        :guide_eta => 1e-3,
        :guide_source_profile => :final,
        :guide_atlas_profile => :final,
        :stages => (:guided, :propagate, :dh_repair,
                    :validate, :edge, :assemble),
        :resume => true,
        :status_only => false,
        :parallel_points => true,
        :parallel_validation => true,
        :gamma_range => nothing,
        :gamma_points => nothing,
        :zeta_range => nothing,
        :zeta_step => nothing,
        :verbosity => 1,
    )
    show_help = false
    for argument in args
        if startswith(argument, "--fingering=")
            values[:fingering] = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--run-id=")
            values[:run_id] = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--profile=")
            values[:profile] = Symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--guide-run-id=")
            values[:guide_run_id] = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--guide-eta=")
            values[:guide_eta] = parse(
                Float64, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--guide-source-profile=")
            values[:guide_source_profile] = Symbol(
                split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--guide-atlas-profile=")
            values[:guide_atlas_profile] = Symbol(
                split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--only=")
            values[:stages] = Tuple(Symbol.(split(
                split(argument, "="; limit=2)[2], ",")))
        elseif startswith(argument, "--gamma-range=")
            values[:gamma_range] = _sax_nonregularized_parse_range(
                split(argument, "="; limit=2)[2], "gamma-range")
        elseif startswith(argument, "--gamma-points=")
            values[:gamma_points] = parse(
                Int, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--zeta-range=")
            values[:zeta_range] = _sax_nonregularized_parse_range(
                split(argument, "="; limit=2)[2], "zeta-range")
        elseif startswith(argument, "--zeta-step=")
            values[:zeta_step] = parse(
                Float64, split(argument, "="; limit=2)[2])
        elseif argument == "--serial-points"
            values[:parallel_points] = false
        elseif argument == "--serial-validation"
            values[:parallel_validation] = false
        elseif argument == "--parallel-validation"
            values[:parallel_validation] = true
        elseif argument == "--status-only"
            values[:status_only] = true
        elseif argument == "--no-resume"
            values[:resume] = false
        elseif startswith(argument, "--verbosity=")
            values[:verbosity] = parse(
                Int, split(argument, "="; limit=2)[2])
        elseif argument in ("-h", "--help")
            show_help = true
        else
            throw(ArgumentError("unknown argument: $(argument)"))
        end
    end
    values[:profile] in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be smoke, pilot, or final"))
    values[:guide_source_profile] in (:smoke, :pilot, :final) ||
        throw(ArgumentError("invalid guide source profile"))
    values[:guide_atlas_profile] in (:smoke, :pilot, :final) ||
        throw(ArgumentError("invalid guide atlas profile"))
    values[:guide_eta] > 0 || throw(ArgumentError(
        "guide eta must be positive"))
    _sax_nonregularized_validate_run_id(values[:run_id])
    _sax_regularized_validate_run_id(values[:guide_run_id])
    allowed = Set((:guided, :propagate, :dh_repair,
                   :validate, :edge, :assemble))
    !isempty(values[:stages]) && all(in(allowed), values[:stages]) ||
        throw(ArgumentError(
            "--only accepts guided,propagate,dh_repair,validate,edge,assemble"))
    isnothing(values[:gamma_points]) || values[:gamma_points] >= 3 ||
        throw(ArgumentError("gamma-points must be at least three"))
    isnothing(values[:zeta_step]) || values[:zeta_step] > 0 ||
        throw(ArgumentError("zeta-step must be positive"))
    values[:verbosity] >= 0 || throw(ArgumentError(
        "verbosity must be nonnegative"))
    return (
        options=SaxNonregularizedMultistabilityPipelineOptions(;
            (name => values[name]
             for name in fieldnames(
                 SaxNonregularizedMultistabilityPipelineOptions))...),
        show_help=show_help,
    )
end

function _sax_nonregularized_pipeline_settings(options)
    overrides = (
        guide_eta=options.guide_eta,
    )
    !isnothing(options.gamma_range) &&
        (overrides = merge(overrides, (gamma_range=options.gamma_range,)))
    !isnothing(options.gamma_points) &&
        (overrides = merge(overrides, (gamma_points=options.gamma_points,)))
    !isnothing(options.zeta_range) &&
        (overrides = merge(overrides, (zeta_range=options.zeta_range,)))
    !isnothing(options.zeta_step) &&
        (overrides = merge(overrides, (zeta_step=options.zeta_step,)))
    return sax_nonregularized_multistability_settings(
        options.profile; overrides...)
end

function _sax_nonregularized_pipeline_context(options)
    settings = _sax_nonregularized_pipeline_settings(options)
    paths = sax_nonregularized_multistability_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        run_id=options.run_id,
        profile=options.profile)
    raw_model = load_object(paths.model)
    guide_paths = sax_regularized_study_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        eta=options.guide_eta,
        profile=options.guide_source_profile,
        run_id=options.guide_run_id)
    guide_model = regularize_sax_model(raw_model; eta=options.guide_eta)
    p2_guide_anchors = sax_nonregularized_p2_guide_anchors(
        guide_paths.p2_r2, guide_model, settings)
    if settings.track_high_p2 && isempty(p2_guide_anchors)
        options.profile == :final && error(
            "the Final exact map requires compatible stable P2 checkpoints below $(guide_paths.p2_r2)")
        @warn(
            "No compatible stable P2 checkpoints were found; this non-Final run can exercise P1 only",
            p2_root=guide_paths.p2_r2)
    end
    guide_signature = sax_nonregularized_guide_signature(
        fingering=options.fingering,
        eta=options.guide_eta,
        run_id=options.guide_run_id,
        source_profile=options.guide_source_profile,
        atlas_profile=options.guide_atlas_profile,
        p2_anchors=p2_guide_anchors)
    seed_source_paths = sax_nonregularized_multistability_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        run_id=SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        profile=:final)
    seed_path = sax_nonregularized_seed_snapshot_path(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        run_id=SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        profile=:final)
    seed_state = load_sax_nonregularized_seed_snapshot(
        seed_path, raw_model;
        settings=sax_nonregularized_multistability_settings(
            :final; guide_eta=options.guide_eta))
    if seed_state.status == :missing && isdir(seed_source_paths.points)
        build_sax_nonregularized_seed_snapshot(
            raw_model, seed_source_paths, seed_path;
            settings=sax_nonregularized_multistability_settings(
                :final; guide_eta=options.guide_eta))
        seed_state = load_sax_nonregularized_seed_snapshot(
            seed_path, raw_model;
            settings=sax_nonregularized_multistability_settings(
                :final; guide_eta=options.guide_eta))
    end
    seed_state.status == :valid || error(
        "no compatible exact P1 seed snapshot exists at $(seed_path): $(seed_state.reason)")
    guide_progress = seed_state.snapshot.progress
    guide_source = (
        status=:valid,
        source=:exact_p1_seed_snapshot,
        model=raw_model,
        path=seed_path,
        counts=seed_state.snapshot.counts,
    )
    return (
        raw_model=raw_model,
        settings=settings,
        paths=paths,
        guide_paths=guide_paths,
        guide_source=guide_source,
        guide_progress=guide_progress,
        seed_path=seed_path,
        p2_guide_anchors=p2_guide_anchors,
        guide_signature=guide_signature,
    )
end

function _sax_nonregularized_log_status(progress, context, options)
    @info(
        "Attracting-response and multistability map status",
        run_id=options.run_id,
        profile=options.profile,
        eta=0.0,
        exact_model=:historical_piecewise_saxRN,
        guide_eta=options.guide_eta,
        guide_role=:initial_states_only,
        p1_seed_source=:validated_exact_atlas_snapshot,
        p1_seed_anchors=context.guide_source.counts.anchors,
        p2_guide_anchors=length(context.p2_guide_anchors),
        status=progress.status,
        point_caches="$(progress.counts.cached_points)/$(progress.counts.expected_points)",
        candidate_low=progress.counts.candidate_low,
        candidate_high=progress.counts.candidate_high,
        validated_low=progress.counts.stable_low,
        validated_high=progress.counts.stable_high,
        validated_p2=progress.counts.stable_p2,
        coexistence=progress.counts.bistable,
        pqz_audit_scheduled=progress.counts.pqz_audit_scheduled,
        pqz_audit_validated=progress.counts.pqz_audit_validated,
        pqz_audit_failed=progress.counts.pqz_audit_failed,
        unresolved_low=progress.counts.unresolved_low,
        pending_validations=progress.counts.validations_pending,
        edge_points="$(progress.counts.edge_points)/$(progress.counts.expected_edge_points)",
        output=context.paths.root,
    )
    return progress
end

function run_sax_nonregularized_multistability_pipeline(;
        options::SaxNonregularizedMultistabilityPipelineOptions=
            SaxNonregularizedMultistabilityPipelineOptions())
    context = _sax_nonregularized_pipeline_context(options)
    mkpath(context.paths.root)
    initial = load_sax_nonregularized_multistability_progress(
        context.raw_model, context.paths, context.guide_signature;
        settings=context.settings)
    _sax_nonregularized_log_status(initial, context, options)
    options.status_only && return (
        success=true, status=:status_only,
        initial=initial, final=initial,
        options=options, context=context)
    manifest = (
        schema_version=SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION,
        analysis=:nonregularized_multistability_pipeline,
        status=:running,
        started_at=Dates.now(),
        options=options,
        settings=_portable_sax_nonregularized_multistability_settings(
            context.settings),
        exact_model=:historical_piecewise_saxRN,
        eta=0.0,
        guide=context.guide_signature,
    )
    _atomic_jld2_save(context.paths.manifest; manifest)
    results = Dict{Symbol,Any}()
    failures = Any[]
    for stage in options.stages
        try
            if stage == :guided
                BLAS.set_num_threads(1)
                results[stage] = compute_sax_nonregularized_guided_points(
                    context.raw_model, context.guide_progress,
                    context.guide_source.model,
                    context.paths, context.guide_signature;
                    p2_guide_anchors=context.p2_guide_anchors,
                    settings=context.settings, resume=options.resume,
                    parallel=options.parallel_points,
                    verbosity=options.verbosity)
            elseif stage == :propagate
                BLAS.set_num_threads(1)
                results[stage] = compute_sax_nonregularized_propagation(
                    context.raw_model, context.paths,
                    context.guide_signature;
                    settings=context.settings, resume=options.resume,
                    parallel=options.parallel_points,
                    verbosity=options.verbosity)
            elseif stage == :dh_repair
                BLAS.set_num_threads(1)
                results[stage] = compute_sax_nonregularized_dh_repair(
                    context.raw_model, context.guide_progress,
                    context.guide_source.model,
                    context.paths, context.guide_signature;
                    settings=context.settings, resume=options.resume,
                    parallel=options.parallel_points,
                    verbosity=options.verbosity)
            elseif stage == :validate
                BLAS.set_num_threads(options.parallel_validation ? 1 :
                                     max(1, Threads.nthreads()))
                results[stage] =
                    compute_sax_nonregularized_attraction_validation(
                        context.raw_model, context.paths,
                        context.guide_signature;
                        settings=context.settings, resume=options.resume,
                        parallel=options.parallel_validation,
                        verbosity=options.verbosity)
            elseif stage == :edge
                BLAS.set_num_threads(1)
                results[stage] = compute_sax_nonregularized_edge_escape(
                    context.raw_model, context.paths,
                    context.guide_signature;
                    settings=context.settings, resume=options.resume,
                    parallel=options.parallel_points,
                    verbosity=options.verbosity)
            elseif stage == :assemble
                results[stage] =
                    assemble_sax_nonregularized_multistability_atlas(
                        context.raw_model, context.paths,
                        context.guide_signature;
                        settings=context.settings,
                        source=(
                            fingering=options.fingering,
                            run_id=options.run_id,
                            guide_run_id=options.guide_run_id,
                            guide_eta=options.guide_eta,
                            guide_source_profile=options.guide_source_profile,
                            guide_atlas_profile=options.guide_atlas_profile,
                        ))
            end
        catch err
            err isa InterruptException && rethrow()
            failure = (
                stage=stage,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err))
            push!(failures, failure)
            @error(
                "Non-regularized multistability stage failed; atomic caches remain reusable",
                failure, exception=(err, catch_backtrace()))
        end
    end
    final = load_sax_nonregularized_multistability_progress(
        context.raw_model, context.paths, context.guide_signature;
        settings=context.settings)
    _sax_nonregularized_log_status(final, context, options)
    requested_complete = all(stage -> haskey(results, stage), options.stages)
    success = isempty(failures) && requested_complete
    final_manifest = merge(manifest, (
        status=success ? :complete : :partial,
        finished_at=Dates.now(),
        failures=failures,
        counts=final.counts))
    _atomic_jld2_save(context.paths.manifest; manifest=final_manifest)
    @info(
        "Non-regularized multistability pipeline finished",
        success, failures=length(failures), output=context.paths.root)
    return (
        success=success,
        status=success ? :complete : :partial,
        results=results,
        failures=failures,
        initial=initial,
        final=final,
        options=options,
        context=context,
    )
end

"""Remote/Windows-friendly REPL entry point with a flushed progress log."""
function run_sax_nonregularized_multistability_unattended(;
        fingering::AbstractString="Dx4",
        run_id::AbstractString=SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        profile::Symbol=:final,
        guide_run_id::AbstractString=SAX_REGULARIZED_CANONICAL_RUN_ID,
        guide_eta::Real=1e-3,
        guide_source_profile::Symbol=:final,
        guide_atlas_profile::Symbol=:final,
        gamma_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        gamma_points::Union{Nothing,Integer}=nothing,
        zeta_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        zeta_step::Union{Nothing,Real}=nothing,
        stages::Tuple{Vararg{Symbol}}=(
            :guided, :propagate, :dh_repair,
            :validate, :edge, :assemble),
        parallel_points::Bool=true,
        parallel_validation::Bool=true,
        verbosity::Integer=1,
        log_path::Union{Nothing,AbstractString}=nothing)
    options = SaxNonregularizedMultistabilityPipelineOptions(
        fingering=String(fingering),
        run_id=_sax_nonregularized_validate_run_id(run_id),
        profile=profile,
        guide_run_id=_sax_regularized_validate_run_id(guide_run_id),
        guide_eta=float(guide_eta),
        guide_source_profile=guide_source_profile,
        guide_atlas_profile=guide_atlas_profile,
        gamma_range=isnothing(gamma_range) ? nothing :
            (float(gamma_range[1]), float(gamma_range[2])),
        gamma_points=isnothing(gamma_points) ? nothing : Int(gamma_points),
        zeta_range=isnothing(zeta_range) ? nothing :
            (float(zeta_range[1]), float(zeta_range[2])),
        zeta_step=isnothing(zeta_step) ? nothing : float(zeta_step),
        stages=stages,
        resume=true,
        parallel_points=parallel_points,
        parallel_validation=parallel_validation,
        verbosity=Int(verbosity))
    context = _sax_nonregularized_pipeline_context(options)
    mkpath(context.paths.root)
    selected_log = isnothing(log_path) ? context.paths.log :
        abspath(String(log_path))
    mkpath(dirname(selected_log))
    println("Exact multistability output: ", context.paths.root)
    println("Progress log: ", selected_log)
    println("Julia threads: ", Threads.nthreads())
    println("Return-map validation: ", parallel_validation ?
            "parallel" : "serial")
    println("Periodic-Schur audit: sparse and serialized")
    flush(stdout)
    result = open(selected_log, "a") do io
        logger = SaxRegularizedFlushLogger(
            ConsoleLogger(io, Logging.Info), io)
        Logging.with_logger(logger) do
            @info(
                "Unattended exact multistability run entered",
                started_at=Dates.now(), options)
            try
                run_sax_nonregularized_multistability_pipeline(
                    options=options)
            finally
                @info(
                    "Unattended exact multistability run left",
                    finished_at=Dates.now())
                flush(io)
            end
        end
    end
    return merge(result, (log_path=selected_log,))
end

"""
Run the independent paper-refinement grid without touching `exact_v1`.

The refined grid halves both spacings, contains every original grid point,
and writes to `exact_v2_refined`. It is intentionally a full exact-model run:
the ordinary map remains available while this restartable calculation is in
progress.
"""
function run_sax_nonregularized_refined_unattended(;
        stages::Tuple{Vararg{Symbol}}=(
            :guided, :propagate, :dh_repair, :validate, :assemble),
        kwargs...)
    return run_sax_nonregularized_multistability_unattended(;
        run_id=SAX_NONREGULARIZED_REFINED_RUN_ID,
        gamma_points=SAX_NONREGULARIZED_REFINED_GAMMA_POINTS,
        zeta_step=SAX_NONREGULARIZED_REFINED_ZETA_STEP,
        stages=stages,
        kwargs...)
end

"""
Run the expanded, paper-resolution exact-model map independently.

The grid extends the refined lattice down to `gamma=0.02` and `zeta=0.0125`
without modifying either `exact_v1` or `exact_v2_refined`.  The lower zeta
bound is positive to avoid the degenerate zero-opening endpoint.  All four
stability/mixed-response figure classes require the edge stage, so it is
included by default; every point and edge cache remains restartable.
"""
function run_sax_nonregularized_expanded_unattended(;
        stages::Tuple{Vararg{Symbol}}=(
            :guided, :propagate, :dh_repair,
            :validate, :edge, :assemble),
        kwargs...)
    return run_sax_nonregularized_multistability_unattended(;
        run_id=SAX_NONREGULARIZED_EXPANDED_RUN_ID,
        gamma_range=SAX_NONREGULARIZED_EXPANDED_GAMMA_RANGE,
        gamma_points=SAX_NONREGULARIZED_EXPANDED_GAMMA_POINTS,
        zeta_range=SAX_NONREGULARIZED_EXPANDED_ZETA_RANGE,
        zeta_step=SAX_NONREGULARIZED_EXPANDED_ZETA_STEP,
        stages=stages,
        kwargs...)
end

Base.@noinline function sax_nonregularized_multistability_main(args=ARGS)
    parsed = try
        _parse_sax_nonregularized_multistability_args(args)
    catch err
        println(stderr, "Error: ", sprint(showerror, err))
        println(stderr)
        println(stderr, _sax_nonregularized_multistability_help())
        return false
    end
    if parsed.show_help
        println(_sax_nonregularized_multistability_help())
        return true
    end
    return Base.invokelatest(
        run_sax_nonregularized_multistability_pipeline;
        options=parsed.options)
end

if abspath(PROGRAM_FILE) == @__FILE__
    outcome = Base.invokelatest(sax_nonregularized_multistability_main)
    failed = outcome === false ||
        (hasproperty(outcome, :success) && !Bool(outcome.success))
    failed && exit(1)
end
