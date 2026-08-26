#!/usr/bin/env julia

"""
Restartable, headless pipeline for the Colinot-regularized saxophone model.

The runner never writes below the historical `bifurcation_Dx4` directory.
Every cache is namespaced by regularization law, eta, and resolution profile.

Examples

    julia --project=. --threads=16 src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl --run-id=canonical_v1 --eta=0.001 --profile=smoke
    julia --project=. --threads=16 src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl --run-id=canonical_v1 --eta=0.001 --profile=final --coverage=all
    julia --project=. --threads=16 src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl --run-id=canonical_v1 --eta=0.0001 --profile=final --coverage=all --main-passes=3 --verify-products
    julia --project=. src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl --run-id=canonical_v1 --eta=0.001 --profile=final --coverage=all --status-only

The default eta is `0.001`. Repeating a command resumes compatible atomic
checkpoints. To test robustness, run the same command with another `--eta`;
the new value automatically receives a different directory.
"""

import Pkg

if !isdefined(@__MODULE__, :SAX_REGULARIZED_PROJECT_ROOT)
    const SAX_REGULARIZED_PROJECT_ROOT =
        normpath(joinpath(@__DIR__, "..", "..", ".."))
end
Pkg.activate(SAX_REGULARIZED_PROJECT_ROOT)

using DifferentialEquations
using Dates
using JLD2
using LinearAlgebra
using Logging
using Plots
using Printf
using Statistics

if !isdefined(@__MODULE__, :set_parameters)
    # Headless includes avoid initializing audio or serial devices on a remote
    # machine. Every dependency has its own guard because a long-lived REPL
    # can contain only part of the analysis stack.
    include(joinpath(SAX_REGULARIZED_PROJECT_ROOT, "src", "sax_model_core.jl"))
end
if !isdefined(@__MODULE__, :SaxRegularizationSettings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis", "sax_regularized_model.jl"))
end
for (sentinel, source) in (
        (:SaxBifurcationSettings, "sax_bifurcation_analysis.jl"),
        (:SaxPDRescueSettings, "sax_pd_rescue_analysis.jl"),
        (:SaxTransitionSettings, "sax_transition_analysis.jl"),
        (:SaxTransitionRefinementSettings,
         "sax_transition_refinement_analysis.jl"),
        (:SaxHighGammaNSSettings, "sax_transition_mechanism_analysis.jl"),
        (:SaxFloquetPQZ, "sax_periodic_schur_ns_analysis.jl"),
        (:SaxDensePQZNSSettings, "sax_dense_pqz_ns_analysis.jl"))
    isdefined(@__MODULE__, sentinel) || include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT, "src", "analysis", source))
end
if !isdefined(@__MODULE__, :SaxFixedZetaAmplitudeSettings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_fixed_zeta_amplitude.jl"))
end
if !isdefined(@__MODULE__, :SaxFixedZetaSchurSettings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_fixed_zeta_schur.jl"))
end
if !isdefined(@__MODULE__, :SaxRegularizedFoldCompletionSettings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_fold_completion.jl"))
end
if !isdefined(@__MODULE__, :SaxRegularizedPlaneCompletionSettings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_plane_completion.jl"))
end
if !isdefined(@__MODULE__, :SaxRegularizedP2Settings)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_p2_analysis.jl"))
end
if !isdefined(@__MODULE__, :SAX_REGULARIZED_CANONICAL_RUN_ID)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_study.jl"))
end
if !isdefined(@__MODULE__, :SAX_REGULARIZED_COMPLETE_PRODUCT_SCHEMA_VERSION)
    include(joinpath(
        SAX_REGULARIZED_PROJECT_ROOT,
        "src", "regularized_bifurcation", "analysis",
        "sax_regularized_complete_product.jl"))
end

if !isdefined(@__MODULE__, :SaxRegularizedPipelineOptions)
    Base.@kwdef struct SaxRegularizedPipelineOptions
        fingering::String = "Dx4"
        run_id::String = SAX_REGULARIZED_CANONICAL_RUN_ID
        eta::Float64 = 1e-3
        profile::Symbol = :pilot
        coverage::Symbol = :focused
        stages::Tuple{Vararg{Symbol}} = (
            :main, :stability, :plane, :folds, :amplitude, :p2, :dense, :assemble)
        reference_eta::Float64 = 1e-3
        include_periodic::Bool = true
        status_only::Bool = false
        resume::Bool = true
        zeta_step::Union{Nothing,Float64} = nothing
        main_passes::Int = 1
        verify_products::Bool = false
        verbosity::Int = 1
    end
end

if !isdefined(@__MODULE__, :SaxRegularizedFlushLogger)
    struct SaxRegularizedFlushLogger{L,I} <: Logging.AbstractLogger
        parent::L
        stream::I
    end
end

Logging.min_enabled_level(logger::SaxRegularizedFlushLogger) =
    Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::SaxRegularizedFlushLogger, args...) =
    Logging.shouldlog(logger.parent, args...)
Logging.catch_exceptions(logger::SaxRegularizedFlushLogger) =
    Logging.catch_exceptions(logger.parent)
function Logging.handle_message(
        logger::SaxRegularizedFlushLogger,
        level,
        message,
        module_value,
        group,
        id,
        file,
        line;
        kwargs...)
    Logging.handle_message(
        logger.parent,
        level,
        message,
        module_value,
        group,
        id,
        file,
        line;
        kwargs...,
    )
    flush(logger.stream)
    return nothing
end

function _sax_regularized_pipeline_help()
    return """
    Usage:
      julia --project=. --threads=16 src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl [options]

    Options:
      --fingering=Dx4                 Acoustic model, default Dx4
      --run-id=canonical_v1           Isolated result namespace
      --eta=0.001                     Colinot regularization value
      --profile=smoke|pilot|final     Resolution profile, default pilot
      --coverage=focused|full|extended|both|all
                                      Dense PQZ coverage; `extended` searches
                                      zeta=0.55:0.01:0.99 up to gamma=0.99
      --only=main,stability,plane,folds,amplitude,p2,dense,assemble
                                      Run selected sequential stages
      --reference-eta=0.001           Use this completed diagram only to match
                                      corresponding PD/R2 components
      --zeta-step=VALUE               Override dense fixed-zeta spacing
      --main-passes=N                 Retry incomplete main components up to N
                                      times, default 1
      --verify-products               Require the duplicate-free product,
                                      including H1/H2/H3, DH, GH, folds,
                                      NS, PD, R2 evidence, and fixed-zeta PQZ
      --equilibria-only               Skip periodic branches in the main stage
      --status-only                   Inspect caches without computing
      --no-resume                     Ignore compatible progress
      --verbosity=N                   Progress verbosity, default 1
      -h, --help                      Show this help

    Recommended publication run:

      julia --project=. --threads=16 src/regularized_bifurcation/scripts/run_regularized_bifurcation_pipeline.jl --run-id=canonical_v1 --eta=0.001 --profile=final --coverage=all --main-passes=3 --verify-products
    """
end

function _parse_sax_regularized_pipeline_args(args)
    values = Dict{Symbol,Any}(
        :fingering => "Dx4",
        :run_id => SAX_REGULARIZED_CANONICAL_RUN_ID,
        :eta => 1e-3,
        :profile => :pilot,
        :coverage => :focused,
        :stages => (
            :main, :stability, :plane, :folds, :amplitude, :p2, :dense, :assemble),
        :reference_eta => 1e-3,
        :include_periodic => true,
        :status_only => false,
        :resume => true,
        :zeta_step => nothing,
        :main_passes => 1,
        :verify_products => false,
        :verbosity => 1,
    )
    show_help = false
    for argument in args
        if startswith(argument, "--fingering=")
            values[:fingering] = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--run-id=")
            values[:run_id] = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--eta=")
            values[:eta] = parse(Float64, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--profile=")
            values[:profile] = Symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--coverage=")
            values[:coverage] = Symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--only=")
            values[:stages] = Tuple(Symbol.(split(
                split(argument, "="; limit=2)[2], ",")))
        elseif startswith(argument, "--reference-eta=")
            values[:reference_eta] = parse(
                Float64, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--zeta-step=")
            values[:zeta_step] = parse(
                Float64, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--main-passes=")
            values[:main_passes] = parse(
                Int, split(argument, "="; limit=2)[2])
        elseif argument == "--verify-products"
            values[:verify_products] = true
        elseif argument == "--equilibria-only"
            values[:include_periodic] = false
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
    values[:eta] > 0 || throw(ArgumentError("eta must be positive"))
    _sax_regularized_validate_run_id(values[:run_id])
    values[:profile] in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be smoke, pilot, or final"))
    values[:coverage] in (:focused, :full, :extended, :both, :all) ||
        throw(ArgumentError(
            "coverage must be focused, full, extended, both, or all"))
    allowed = Set((
        :main, :stability, :plane, :folds, :amplitude, :p2, :dense, :assemble))
    !isempty(values[:stages]) && all(in(allowed), values[:stages]) ||
        throw(ArgumentError(
            "--only accepts main,stability,plane,folds,amplitude,p2,dense,assemble"))
    values[:reference_eta] > 0 || throw(ArgumentError(
        "reference-eta must be positive"))
    values[:verbosity] >= 0 || throw(ArgumentError(
        "verbosity must be nonnegative"))
    values[:main_passes] >= 1 || throw(ArgumentError(
        "main-passes must be at least one"))
    (isnothing(values[:zeta_step]) || values[:zeta_step] > 0) ||
        throw(ArgumentError("zeta-step must be positive"))
    return (
        options=SaxRegularizedPipelineOptions(;
            fingering=values[:fingering],
            run_id=values[:run_id],
            eta=values[:eta],
            profile=values[:profile],
            coverage=values[:coverage],
            stages=values[:stages],
            reference_eta=values[:reference_eta],
            include_periodic=values[:include_periodic],
            status_only=values[:status_only],
            resume=values[:resume],
            zeta_step=values[:zeta_step],
            main_passes=values[:main_passes],
            verify_products=values[:verify_products],
            verbosity=values[:verbosity],
        ),
        show_help=show_help,
    )
end

function _sax_regularized_coverages(coverage::Symbol)
    return coverage == :both ? (:focused, :full) :
        coverage == :all ? (:focused, :full, :extended) : (coverage,)
end

function _sax_regularized_dense_paths(paths, coverage::Symbol)
    directory = getproperty(paths, Symbol("dense_$(coverage)"))
    return (
        directory=directory,
        slices=joinpath(directory, "slices"),
        validation=joinpath(directory, "seed_validation.jld2"),
        seed=joinpath(directory, "refined_seed.jld2"),
        augmented=joinpath(directory, "augmented"),
    )
end

function _sax_regularized_manifest!(paths, model, options; status=:initialized)
    manifest = (
        schema_version=2,
        status=status,
        saved_at_unix=time(),
        fingering=options.fingering,
        run_id=options.run_id,
        profile=options.profile,
        coverage=options.coverage,
        stages=options.stages,
        include_periodic=options.include_periodic,
        main_passes=options.main_passes,
        reference_eta=options.reference_eta,
        verify_products=options.verify_products,
        regularization=model.sax_regularization,
        model_signature=_sax_bifurcation_model_signature(model, 8),
        source=(
            author="Tom Colinot",
            title="Numerical simulation of woodwind dynamics",
            year=2020,
            law="abs(z) approximately sqrt(z^2 + eta)",
        ),
    )
    _atomic_jld2_save(paths.manifest; manifest)
    return manifest
end

function _sax_regularized_stage_keys_complete(
        loaded,
        completed_field::Symbol,
        expected_field::Symbol)
    loaded.status == :valid || return false
    completed = Set(String.(getproperty(loaded.payload, completed_field)))
    expected = hasproperty(loaded.payload, expected_field) ?
        Set(String.(getproperty(loaded.payload, expected_field))) : completed
    return issubset(expected, completed)
end

"""Inspect whether every restartable main stage and the portable result agree."""
function _sax_regularized_main_completion(model, paths, options)
    settings = sax_regularized_bifurcation_settings(options.profile)
    main = load_sax_bifurcation_cache(
        paths.main, model;
        fingering=options.fingering,
        settings=settings,
    )
    hopf = _load_sax_stage_cache(
        _sax_stage_cache_path(paths.stages, :hopf),
        :hopf, model, settings)
    periodic = _load_sax_stage_cache(
        _sax_stage_cache_path(paths.stages, :periodic),
        :periodic, model, settings)
    curves = _load_sax_stage_cache(
        _sax_stage_cache_path(paths.stages, :curves),
        :curves, model, settings)

    hopf_complete = hopf.status == :valid &&
        length(hopf.payload.completed_seeds) >= length(settings.zeta_seeds)
    periodic_complete = _sax_regularized_stage_keys_complete(
        periodic, :completed_hopf_keys, :expected_hopf_keys)
    expected_curve_keys = periodic.status == :valid ?
        Set(String(checkpoint.key) for checkpoint in
            periodic.payload.periodic_checkpoints) : Set{String}()
    completed_curve_keys = curves.status == :valid ?
        Set(String.(curves.payload.completed_checkpoint_keys)) : Set{String}()
    curves_complete = periodic_complete &&
        (isempty(expected_curve_keys) ||
         (curves.status == :valid &&
          issubset(expected_curve_keys, completed_curve_keys)))

    synchronized = false
    if main.status == :valid && periodic.status == :valid &&
            (curves.status == :valid || isempty(expected_curve_keys))
        result = main.result
        fold_count = curves.status == :valid ?
            length(curves.payload.fold_curves) : 0
        pd_count = curves.status == :valid ? length(curves.payload.pd_curves) : 0
        ns_count = curves.status == :valid ? length(curves.payload.ns_curves) : 0
        synchronized = hasproperty(result.counts, :periodic_branches) &&
            result.counts.periodic_branches ==
                periodic.payload.periodic_branch_count &&
            result.counts.fold_curves == fold_count &&
            result.counts.pd_curves == pd_count &&
            result.counts.ns_curves == ns_count
    end
    complete = main.status == :valid && hopf_complete &&
        (!options.include_periodic ||
         (periodic_complete && curves_complete && synchronized))
    return (
        complete=complete,
        main_status=main.status,
        hopf_complete=hopf_complete,
        periodic_complete=periodic_complete,
        curves_complete=curves_complete,
        synchronized=synchronized,
        completed_curve_components=length(completed_curve_keys),
        expected_curve_components=length(expected_curve_keys),
    )
end

Base.@noinline function _log_sax_regularized_state(model, paths, options)
    settings = sax_regularized_bifurcation_settings(options.profile)
    main = load_sax_bifurcation_cache(
        paths.main, model;
        fingering=options.fingering,
        settings=settings,
    )
    @info "Regularized main cache" status=main.status reason=main.reason path=paths.main
    stages = sax_bifurcation_stage_cache_status(
        paths.stages, model; settings=settings)
    for stage in stages
        @info "Regularized main checkpoint" stage=stage.stage status=stage.status completed=stage.completed total=stage.total reason=stage.reason
    end
    dense = Dict{Symbol,Any}()
    for coverage in _sax_regularized_coverages(options.coverage)
        selected_paths = _sax_regularized_dense_paths(paths, coverage)
        settings_dense = sax_dense_pqz_cached_settings(
            selected_paths.directory;
            fallback=_sax_regularized_dense_settings(
                options.profile, coverage; zeta_step=options.zeta_step),
        )
        progress = load_sax_dense_pqz_ns_progress(
            model, selected_paths.directory; settings=settings_dense)
        dense[coverage] = progress
        @info "Regularized dense PQZ cache" coverage slices=progress.slices.status completed=progress.slices.completed expected=progress.slices.expected roots=length(progress.slices.roots) accepted_roots=progress.slices.accepted_roots augmented=progress.augmented.status augmented_curves=length(progress.augmented.curves)
    end
    p2_settings = sax_regularized_p2_settings(
        options.profile; reference_eta=options.reference_eta)
    p2 = load_sax_regularized_p2_progress(
        model, paths.root; settings=p2_settings)
    @info(
        "Regularized P2/R2 Periodic-Schur cache",
        status=p2.status,
        source=p2.source_kind,
        completed_components=count(component -> component.status == :complete,
                                   p2.components),
        expected_components=p2.task_count,
        p2_curves=length(p2.p2_curves),
        p2_ns_roots=count(root -> root.accepted, p2.ns_roots),
    )
    amplitude_settings = sax_fixed_zeta_amplitude_settings(options.profile)
    amplitude = load_sax_fixed_zeta_amplitude_progress(
        model, paths.root; settings=amplitude_settings)
    @info(
        "Regularized fixed-zeta amplitude cache",
        status=amplitude.status,
        zeta=amplitude.settings.zeta,
        p1="$(amplitude.counts.p1)/$(amplitude.counts.expected_p1)",
        p2="$(amplitude.counts.p2)/$(amplitude.counts.expected_p2)",
        output=amplitude.paths.directory,
    )
    stability_settings = sax_fixed_zeta_schur_settings(options.profile)
    stability = load_sax_fixed_zeta_schur_progress(
        model, paths.root; settings=stability_settings)
    @info(
        "Regularized fixed-zeta Periodic-Schur cache",
        status=stability.status,
        zeta=stability.settings.zeta,
        components="$(stability.counts.components)/$(stability.counts.expected_components)",
        samples=stability.counts.samples,
        events=stability.counts.events,
        validated_events=stability.counts.validated_events,
        output=stability.paths.directory,
    )
    plane_settings = sax_regularized_plane_completion_settings(options.profile)
    plane = load_sax_regularized_plane_completion_progress(
        model, paths.root;
        settings=plane_settings,
        stability_settings=stability_settings,
    )
    @info(
        "Regularized Periodic-Schur seeded plane curves",
        status=plane.status,
        source_slices="$(plane.counts.source_complete)/$(plane.counts.source_expected)",
        components="$(plane.counts.complete)/$(plane.counts.expected)",
        partial=plane.counts.partial,
        pd_curves=plane.counts.pd,
        ns_curves=plane.counts.ns,
        output=plane.paths.directory,
    )
    fold_settings = sax_regularized_fold_completion_settings(options.profile)
    folds = load_sax_regularized_fold_completion_progress(
        model, paths.root; settings=fold_settings)
    @info(
        "Regularized fold-completion cache",
        status=folds.status,
        components="$(folds.counts.complete)/$(folds.counts.expected)",
        partial=folds.counts.partial,
        curves=folds.counts.curves,
        dual_floquet_validated=folds.counts.dual_floquet_validated,
        output=folds.paths.directory,
    )
    complete_product = load_sax_regularized_complete_product(
        model,
        paths;
        fingering=options.fingering,
        profile=options.profile,
        coverages=_sax_regularized_coverages(options.coverage),
        zeta_step=options.zeta_step,
        reference_eta=options.reference_eta,
    )
    @info(
        "Regularized canonical product",
        status=complete_product.status,
        scientific_status=complete_product.status == :valid ?
            complete_product.product.status : :unavailable,
        missing=complete_product.status == :valid ?
            complete_product.product.validation.missing : Symbol[],
        path=complete_product.path,
    )
    return (main=main, stages=stages, dense=dense, p2=p2,
            amplitude=amplitude, stability=stability, plane=plane, folds=folds,
            complete_product=complete_product)
end

Base.@noinline function _run_sax_regularized_main!(model, paths, options)
    settings = sax_regularized_bifurcation_settings(options.profile)
    cached = load_sax_bifurcation_cache(
        paths.main, model;
        fingering=options.fingering,
        settings=settings,
    )
    completion = _sax_regularized_main_completion(model, paths, options)
    if cached.status == :valid &&
            (!options.include_periodic || completion.complete)
        @info "Skipping compatible regularized main cache" path=paths.main
        return cached.result
    end
    cached.status == :valid && options.include_periodic && @info(
        "A compatible partial main cache exists; resuming incomplete stages",
        completion,
    )
    result = Base.invokelatest(
        compute_sax_bifurcation_diagram,
        model;
        settings=settings,
        include_periodic=options.include_periodic,
        verbosity=options.verbosity,
        stage_cache_directory=paths.stages,
        resume=options.resume,
        periodic_eigsolver=SaxFloquetPQZ(
            cyclic_retries=8,
            fallback_to_floquet_coll=false,
        ),
    )
    return save_sax_bifurcation_cache(
        paths.main, result, model; fingering=options.fingering)
end

function _sax_regularized_is_r2_evidence(point)
    hasproperty(point, :classification) &&
        point.classification in (:near_r2, :resonant_1_2) && return true
    hasproperty(point, :near_r2) && point.near_r2 && return true
    hasproperty(point, :root) &&
        _sax_regularized_is_r2_evidence(point.root) && return true
    hasproperty(point, :periodic_schur) &&
        _sax_regularized_is_r2_evidence(point.periodic_schur) && return true
    return false
end

Base.@noinline function _run_sax_regularized_p2!(model, paths, options)
    settings = sax_regularized_p2_settings(
        options.profile; reference_eta=options.reference_eta)
    reference = isapprox(options.eta, options.reference_eta; rtol=0, atol=0) ?
        nothing : sax_regularized_study_paths(
            SAX_REGULARIZED_PROJECT_ROOT;
            fingering=options.fingering,
            eta=options.reference_eta,
            profile=options.profile,
            run_id=options.run_id,
        ).stages
    fixed_zeta_progress = load_sax_fixed_zeta_schur_progress(
        model, paths.root;
        settings=sax_fixed_zeta_schur_settings(options.profile),
    )
    plane_progress = load_sax_regularized_plane_completion_progress(
        model, paths.root;
        settings=sax_regularized_plane_completion_settings(options.profile),
        stability_settings=sax_fixed_zeta_schur_settings(options.profile),
    )
    return Base.invokelatest(
        compute_sax_regularized_p2_r2,
        model,
        paths.stages,
        paths.root;
        main_settings=sax_regularized_bifurcation_settings(options.profile),
        settings=settings,
        reference_stage_directory=reference,
        fixed_zeta_progress=fixed_zeta_progress,
        plane_progress=plane_progress,
        resume=options.resume,
        verbosity=options.verbosity,
    )
end

Base.@noinline function _run_sax_regularized_amplitude!(model, paths, options)
    settings = sax_fixed_zeta_amplitude_settings(options.profile)
    return Base.invokelatest(
        compute_sax_fixed_zeta_amplitude,
        model,
        paths.root;
        settings=settings,
        resume=options.resume,
        verbosity=options.verbosity,
    )
end

Base.@noinline function _run_sax_regularized_stability!(model, paths, options)
    settings = sax_fixed_zeta_schur_settings(options.profile)
    return Base.invokelatest(
        compute_sax_fixed_zeta_schur,
        model,
        paths.root;
        settings=settings,
        resume=options.resume,
        verbosity=options.verbosity,
    )
end

Base.@noinline function _run_sax_regularized_plane!(model, paths, options)
    settings = sax_regularized_plane_completion_settings(options.profile)
    stability_settings = sax_fixed_zeta_schur_settings(options.profile)
    return Base.invokelatest(
        compute_sax_regularized_plane_completion,
        model,
        paths.root;
        settings=settings,
        stability_settings=stability_settings,
        resume=options.resume,
        verbosity=options.verbosity,
    )
end

Base.@noinline function _run_sax_regularized_folds!(model, paths, options)
    settings = sax_regularized_fold_completion_settings(options.profile)
    return Base.invokelatest(
        compute_sax_regularized_fold_completion,
        model,
        paths.root;
        settings=settings,
        resume=options.resume,
        verbosity=options.verbosity,
    )
end

Base.@noinline function _run_sax_regularized_assemble!(model, paths, options)
    return Base.invokelatest(
        compute_sax_regularized_complete_product,
        model,
        paths;
        fingering=options.fingering,
        profile=options.profile,
        coverages=_sax_regularized_coverages(options.coverage),
        zeta_step=options.zeta_step,
        reference_eta=options.reference_eta,
    )
end

"""Check the products requested for a publication-resolution robustness run."""
function audit_sax_regularized_bifurcation_products(model, paths, options)
    assembled = load_sax_regularized_complete_product(
        model,
        paths;
        fingering=options.fingering,
        profile=options.profile,
        coverages=_sax_regularized_coverages(options.coverage),
        zeta_step=options.zeta_step,
        reference_eta=options.reference_eta,
    )
    if assembled.status == :valid
        product = assembled.product
        checks = merge(
            (complete_product=product.status == :complete,),
            product.validation.checks,
        )
        missing = Symbol[name for (name, passed) in pairs(checks) if !passed]
        return (
            passed=isempty(missing),
            checks=checks,
            missing=missing,
            product_status=product.status,
            product_path=assembled.path,
            counts=product.counts,
            duplicates=product.duplicates,
        )
    end
    settings = sax_regularized_bifurcation_settings(options.profile)
    main = load_sax_bifurcation_cache(
        paths.main, model;
        fingering=options.fingering,
        settings=settings,
    )
    result = main.status == :valid ? main.result : nothing
    hopf_modes = isnothing(result) ? Int[] : sort(unique(
        Int(curve.mode) for curve in result.hopf_curves if curve.mode > 0))
    valid_dh = isnothing(result) ? 0 : count(point -> point.valid,
        result.double_hopf_points)
    valid_gh = isnothing(result) ? 0 : count(point -> point.valid,
        result.generalized_hopf_points)
    pd_curves = isnothing(result) ? 0 : length(result.pd_curves)
    ns_curves = isnothing(result) ? 0 : length(result.ns_curves)
    high_gamma_ns = !isnothing(result) && any(
        curve -> !isempty(curve.gamma) && maximum(curve.gamma) >= 0.60,
        result.ns_curves)
    p2_settings = sax_regularized_p2_settings(
        options.profile; reference_eta=options.reference_eta)
    p2 = load_sax_regularized_p2_progress(
        model, paths.root; settings=p2_settings)
    p2_components = count(component -> component.status == :complete,
                          p2.components)
    p2_pqz_samples = sum((
        component.status == :complete ? component.analysis.total_samples : 0
        for component in p2.components); init=0)

    r2_evidence = 0
    for coverage in _sax_regularized_coverages(options.coverage)
        selected_paths = _sax_regularized_dense_paths(paths, coverage)
        dense_settings = sax_dense_pqz_cached_settings(
            selected_paths.directory;
            fallback=_sax_regularized_dense_settings(
                options.profile, coverage; zeta_step=options.zeta_step),
        )
        progress = load_sax_dense_pqz_ns_progress(
            model, selected_paths.directory; settings=dense_settings)
        if hasproperty(progress, :slices)
            for field in (:provisional_brackets, :roots)
                hasproperty(progress.slices, field) || continue
                r2_evidence += count(
                    _sax_regularized_is_r2_evidence,
                    getproperty(progress.slices, field),
                )
            end
        end
    end
    completion = _sax_regularized_main_completion(model, paths, options)
    checks = (
        complete_product=false,
        main_complete=completion.complete,
        h1=1 in hopf_modes,
        h2=2 in hopf_modes,
        h3=3 in hopf_modes,
        double_hopf=valid_dh >= 1,
        generalized_hopf=valid_gh >= 1,
        period_doubling=pd_curves >= 1,
        two_ns_components=ns_curves >= 2,
        high_gamma_ns=high_gamma_ns,
        r2_evidence=r2_evidence >= 1,
        p2_periodic_schur=p2_components >= 1 && p2_pqz_samples >= 1,
    )
    missing = Symbol[name for (name, passed) in pairs(checks) if !passed]
    return (
        passed=isempty(missing),
        checks=checks,
        missing=missing,
        hopf_modes=hopf_modes,
        valid_double_hopf_points=valid_dh,
        valid_generalized_hopf_points=valid_gh,
        pd_curves=pd_curves,
        ns_curves=ns_curves,
        r2_evidence=r2_evidence,
        p2_components=p2_components,
        p2_pqz_samples=p2_pqz_samples,
        completion=completion,
        product_status=assembled.status,
        product_path=assembled.path,
    )
end

Base.@noinline function _run_sax_regularized_dense!(
        model, paths, options, coverage::Symbol)
    selected_paths = _sax_regularized_dense_paths(paths, coverage)
    mkpath(selected_paths.directory)
    requested = _sax_regularized_dense_settings(
        options.profile, coverage; zeta_step=options.zeta_step)
    settings = options.resume && isnothing(options.zeta_step) ?
        sax_dense_pqz_cached_settings(
            selected_paths.directory; fallback=requested) : requested
    legacy = _sax_dense_legacy_settings(settings)

    # Validation is informative and may fail when the main stage has not yet
    # produced a compatible Hopf checkpoint. It does not block event-free PQZ.
    try
        Base.invokelatest(
            compute_sax_periodic_schur_seed_validation,
            model, paths.stages, selected_paths.validation;
            settings=legacy,
            resume=options.resume,
            verbosity=options.verbosity,
        )
    catch err
        err isa InterruptException && rethrow()
        @warn "Regularized Periodic-Schur seed validation was unavailable; continuing with independent fixed-zeta discovery" exception=(err, catch_backtrace())
    end

    slices = Base.invokelatest(
        compute_sax_dense_pqz_ns_slices,
        model, selected_paths.slices;
        settings=settings,
        resume=options.resume,
        verbosity=options.verbosity,
    )
    seed = select_sax_dense_pqz_ns_seed(
        model, slices, selected_paths.seed; settings=settings)
    near_r2_seed = seed.status == :valid &&
        hasproperty(seed, :root) &&
        hasproperty(seed.root, :near_r2) &&
        seed.root.near_r2
    if near_r2_seed
        @info "Skipping generic augmented NS continuation for a near-1:2 root" gamma=seed.seed.gamma zeta=seed.seed.zeta theta=seed.seed.floquet_angle
        no_ns_seed = (
            analysis=:dense_pqz_ns_seed,
            status=:missing,
            seed=nothing,
            reason="accepted neutral root is near 1:2 resonance and is not a generic NS seed",
        )
        Base.invokelatest(
            compute_sax_periodic_schur_augmented_ns,
            model, no_ns_seed, selected_paths.augmented;
            settings=legacy,
            resume=false,
            verbosity=options.verbosity,
        )
    elseif seed.status == :valid
        seed = sax_dense_pqz_augmented_seed(seed, settings)
        _save_sax_transition_mechanism_cache(
            selected_paths.seed, :dense_pqz_ns_seed,
            seed, model, settings)
        try
            Base.invokelatest(
                compute_sax_periodic_schur_augmented_ns,
                model, seed, selected_paths.augmented;
                settings=legacy,
                resume=options.resume,
                verbosity=options.verbosity,
            )
        catch err
            err isa InterruptException && rethrow()
            @warn "Regularized augmented NS continuation failed; fixed-zeta PQZ checkpoints remain usable" exception=(err, catch_backtrace())
        end
    else
        @warn "No accepted regularized NS seed is available for augmented continuation" coverage seed_status=seed.status
    end
    return load_sax_dense_pqz_ns_progress(
        model, selected_paths.directory; settings=settings)
end

"""Read regularized cache status without compiling any continuation stage."""
Base.@noinline function inspect_sax_regularized_bifurcation_pipeline(
        options::SaxRegularizedPipelineOptions)
    paths = sax_regularized_study_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        eta=options.eta,
        profile=options.profile,
        run_id=options.run_id,
    )
    raw_model = load_object(paths.model)
    model = regularize_sax_model(raw_model; eta=options.eta)
    BLAS.set_num_threads(max(1, Threads.nthreads()))
    @info "Regularized saxophone bifurcation pipeline status" fingering=options.fingering run_id=options.run_id eta=options.eta profile=options.profile coverage=options.coverage julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads() output=paths.root
    state = Base.invokelatest(
        _log_sax_regularized_state, model, paths, options)
    return (
        success=true,
        status=:status_only,
        initial=state,
        final=state,
        paths=paths,
    )
end

"""Run the selected regularized stages sequentially and preserve checkpoints."""
function run_sax_regularized_bifurcation_pipeline(;
        options::SaxRegularizedPipelineOptions=
            SaxRegularizedPipelineOptions())
    paths = sax_regularized_study_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        eta=options.eta,
        profile=options.profile,
        run_id=options.run_id,
    )
    raw_model = load_object(paths.model)
    model = regularize_sax_model(raw_model; eta=options.eta)
    BLAS.set_num_threads(max(1, Threads.nthreads()))
    @info "Regularized saxophone bifurcation pipeline" fingering=options.fingering run_id=options.run_id eta=options.eta profile=options.profile coverage=options.coverage stages=options.stages julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads() output=paths.root
    initial = Base.invokelatest(
        _log_sax_regularized_state, model, paths, options)
    mkpath(paths.root)
    _sax_regularized_manifest!(paths, model, options; status=:running)

    failures = Any[]
    results = Dict{Symbol,Any}()
    for stage in options.stages
        if stage == :main
            main_complete = false
            last_main_error = nothing
            for pass in 1:options.main_passes
                try
                    results[:main] = Base.invokelatest(
                        _run_sax_regularized_main!, model, paths, options)
                    last_main_error = nothing
                catch err
                    err isa InterruptException && rethrow()
                    last_main_error = sprint(showerror, err)
                    @warn "Regularized main pass failed; checkpoints remain reusable" pass pass_limit=options.main_passes exception=(err, catch_backtrace())
                end
                completion = _sax_regularized_main_completion(
                    model, paths, options)
                if completion.complete
                    main_complete = true
                    @info "Regularized main stage is complete" pass completion
                    break
                end
                pass < options.main_passes && @warn(
                    "Regularized main stage remains incomplete; retrying compatible checkpoints",
                    pass,
                    pass_limit=options.main_passes,
                    completion,
                )
            end
            if !main_complete
                completion = _sax_regularized_main_completion(
                    model, paths, options)
                push!(failures, (
                    stage=:main_incomplete,
                    error=isnothing(last_main_error) ?
                        "main checkpoint set remains incomplete" : last_main_error,
                    completion=completion,
                ))
                @error "Regularized main stage exhausted its retry passes" completion
            end
        elseif stage == :p2
            try
                results[:p2] = Base.invokelatest(
                    _run_sax_regularized_p2!, model, paths, options)
                if results[:p2].status != :complete
                    push!(failures, (
                        stage=:p2_incomplete,
                        error="one or more R2-neighbour P2 components failed",
                        captured_failures=results[:p2].failures,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:p2,
                    error=sprint(showerror, err),
                ))
                @error "Regularized P2/R2 stage failed; component caches remain reusable" exception=(err, catch_backtrace())
            end
        elseif stage == :amplitude
            try
                results[:amplitude] = Base.invokelatest(
                    _run_sax_regularized_amplitude!, model, paths, options)
                if results[:amplitude].status != :complete
                    push!(failures, (
                        stage=:amplitude_incomplete,
                        error="one or more fixed-zeta amplitude components are missing",
                        counts=results[:amplitude].counts,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:amplitude,
                    error=sprint(showerror, err),
                ))
                @error "Regularized fixed-zeta amplitude stage failed; component caches remain reusable" exception=(err, catch_backtrace())
            end
        elseif stage == :stability
            try
                results[:stability] = Base.invokelatest(
                    _run_sax_regularized_stability!, model, paths, options)
                if results[:stability].status != :complete
                    push!(failures, (
                        stage=:stability_incomplete,
                        error="one or more fixed-zeta Periodic-Schur components are missing",
                        counts=results[:stability].counts,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:stability,
                    error=sprint(showerror, err),
                ))
                @error "Regularized fixed-zeta Periodic-Schur stage failed; component caches remain reusable" exception=(err, catch_backtrace())
            end
        elseif stage == :plane
            try
                for completion_pass in 1:2
                    results[:plane] = Base.invokelatest(
                        _run_sax_regularized_plane!, model, paths, options)
                    results[:plane].status == :complete && break
                    completion_pass < 2 && @info(
                        "Regularized plane stage remains partial; retrying restartable components",
                        completion_pass,
                        counts=results[:plane].counts,
                    )
                end
                if results[:plane].status != :complete
                    push!(failures, (
                        stage=:plane_incomplete,
                        error="one or more fixed-zeta seeded PD/NS curves are incomplete",
                        counts=results[:plane].counts,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:plane,
                    error=sprint(showerror, err),
                ))
                @error "Regularized Periodic-Schur seeded plane stage failed; component checkpoints remain reusable" exception=(err, catch_backtrace())
            end
        elseif stage == :folds
            try
                for completion_pass in 1:2
                    results[:folds] = Base.invokelatest(
                        _run_sax_regularized_folds!, model, paths, options)
                    results[:folds].status == :complete && break
                    completion_pass < 2 && @info(
                        "Regularized fold stage remains partial; retrying restartable components",
                        completion_pass,
                        counts=results[:folds].counts,
                    )
                end
                if results[:folds].status != :complete
                    push!(failures, (
                        stage=:folds_incomplete,
                        error="one or more regularized fold components are incomplete",
                        counts=results[:folds].counts,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:folds,
                    error=sprint(showerror, err),
                ))
                @error "Regularized fold-completion stage failed; component checkpoints remain reusable" exception=(err, catch_backtrace())
            end
        elseif stage == :dense
            for coverage in _sax_regularized_coverages(options.coverage)
                try
                    results[Symbol("dense_$(coverage)")] =
                        Base.invokelatest(
                            _run_sax_regularized_dense!,
                            model, paths, options, coverage)
                catch err
                    err isa InterruptException && rethrow()
                    push!(failures, (
                        stage=:dense, coverage=coverage,
                        error=sprint(showerror, err)))
                    @error "Regularized dense PQZ stage failed; checkpoints remain reusable" coverage exception=(err, catch_backtrace())
                end
            end
        elseif stage == :assemble
            try
                results[:assemble] = Base.invokelatest(
                    _run_sax_regularized_assemble!, model, paths, options)
                if results[:assemble].status != :complete
                    push!(failures, (
                        stage=:assemble_incomplete,
                        error="one or more canonical bifurcation products are missing",
                        missing=results[:assemble].validation.missing,
                    ))
                end
            catch err
                err isa InterruptException && rethrow()
                push!(failures, (
                    stage=:assemble,
                    error=sprint(showerror, err),
                ))
                @error "Regularized canonical assembly failed; source caches remain reusable" exception=(err, catch_backtrace())
            end
        else
            error("unhandled regularized pipeline stage $(stage)")
        end
    end
    final = Base.invokelatest(
        _log_sax_regularized_state, model, paths, options)
    audit = audit_sax_regularized_bifurcation_products(
        model, paths, options)
    if options.verify_products && !audit.passed
        push!(failures, (
            stage=:product_audit,
            error="required regularized products are missing",
            missing=audit.missing,
        ))
        @error "Regularized product audit did not pass" audit
    else
        @info "Regularized product audit" audit
    end
    success = isempty(failures)
    _sax_regularized_manifest!(
        paths, model, options; status=success ? :complete : :partial)
    @info "Regularized saxophone bifurcation pipeline finished" success failures=length(failures) output=paths.root
    return (
        success=success,
        status=success ? :complete : :partial,
        results=results,
        failures=failures,
        initial=initial,
        final=final,
        audit=audit,
        paths=paths,
    )
end

"""
    run_sax_regularized_unattended(; eta=1e-3, ...)

Run the publication-resolution regularized study from an interactive Julia
REPL, append progress to a file inside the eta-specific result directory, retry
incomplete main components, and require the expected curve families. This is
the recommended entry point for a remote Windows machine.
"""
function run_sax_regularized_unattended(;
        fingering::AbstractString="Dx4",
        run_id::AbstractString=SAX_REGULARIZED_CANONICAL_RUN_ID,
        eta::Real=1e-3,
        profile::Symbol=:final,
        coverage::Symbol=:all,
        main_passes::Integer=3,
        verbosity::Integer=1,
        log_path::Union{Nothing,AbstractString}=nothing)
    options = SaxRegularizedPipelineOptions(
        fingering=String(fingering),
        run_id=_sax_regularized_validate_run_id(run_id),
        eta=float(eta),
        profile=profile,
        coverage=coverage,
        stages=(
            :main, :stability, :plane, :folds, :amplitude, :p2, :dense, :assemble),
        reference_eta=1e-3,
        include_periodic=true,
        status_only=false,
        resume=true,
        main_passes=Int(main_passes),
        verify_products=true,
        verbosity=Int(verbosity),
    )
    paths = sax_regularized_study_paths(
        SAX_REGULARIZED_PROJECT_ROOT;
        fingering=options.fingering,
        eta=options.eta,
        profile=options.profile,
        run_id=options.run_id,
    )
    mkpath(paths.root)
    selected_log = isnothing(log_path) ?
        joinpath(paths.root, "unattended_pipeline.log") :
        abspath(String(log_path))
    mkpath(dirname(selected_log))
    println("Regularized run output: ", paths.root)
    println("Progress log: ", selected_log)
    println("Julia threads: ", Threads.nthreads(),
            "; BLAS threads will be set to the same value")
    flush(stdout)
    result = open(selected_log, "a") do io
        logger = SaxRegularizedFlushLogger(
            ConsoleLogger(io, Logging.Info), io)
        Logging.with_logger(logger) do
            @info "Unattended regularized run entered" started_at=Dates.now() options output=paths.root
            try
                run_sax_regularized_bifurcation_pipeline(options=options)
            finally
                @info "Unattended regularized run left" finished_at=Dates.now()
                flush(io)
            end
        end
    end
    return merge(result, (log_path=selected_log,))
end

Base.@noinline function sax_regularized_bifurcation_main(args=ARGS)
    parsed = try
        _parse_sax_regularized_pipeline_args(args)
    catch err
        println(stderr, "Error: ", sprint(showerror, err))
        println(stderr)
        println(stderr, _sax_regularized_pipeline_help())
        return false
    end
    if parsed.show_help
        println(_sax_regularized_pipeline_help())
        return true
    end
    if parsed.options.status_only
        return Base.invokelatest(
            inspect_sax_regularized_bifurcation_pipeline,
            parsed.options,
        )
    end
    return Base.invokelatest(
        run_sax_regularized_bifurcation_pipeline;
        options=parsed.options,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    outcome = Base.invokelatest(sax_regularized_bifurcation_main)
    failed = outcome === false ||
        (hasproperty(outcome, :success) && !Bool(outcome.success))
    failed && exit(1)
end
