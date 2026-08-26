# Two-parameter PD and NS completion from fixed-zeta Periodic-Schur
# checkpoints. This is the canonical bridge between the stability audit and
# BifurcationKit's minimally augmented curve continuations.

const SAX_REGULARIZED_PLANE_COMPLETION_SCHEMA_VERSION = 4

Base.@kwdef struct SaxRegularizedPlaneCompletionSettings
    schema_version::Int = SAX_REGULARIZED_PLANE_COMPLETION_SCHEMA_VERSION
    nmodes::Int = 8
    seed_zeta::Float64 = 0.6
    source_slices::Tuple = (
        (zeta=0.25, modes=(2,),
         hopf_gamma_hints=(0.4318585085,), directions=(:positive,)),
        # H1 is sampled below the former zeta≈0.42 interruption.  Starting
        # the augmented mode-1 NS continuation here lets its negative arm
        # terminate at the H1/H2 double-Hopf point instead of relying on a
        # distant zeta=0.55 seed to cross the difficult segment.
        (zeta=0.35, modes=(1,),
         hopf_gamma_hints=(0.4020,), directions=(:positive,)),
        (zeta=0.55, modes=(1, 2, 3),
         hopf_gamma_hints=(0.3779789703, 0.3928014803, 0.6391400573),
         directions=(:positive,)),
        (zeta=0.85, modes=(1, 3),
         hopf_gamma_hints=(0.3647019865, 0.5977503219),
         directions=(:positive, :negative)),
    )
    # The lower mode-2 arm leaving the H1/H2 double-Hopf point is already
    # close to the strong 1:2 limit at zeta=0.25.  The fixed-slice classifier
    # therefore calls this neutral complex-pair root :r2 rather than generic
    # :ns.  Only the low-gamma root in this narrow, explicit window is allowed
    # to seed NS continuation; the second high-gamma near-R2 root remains R2
    # evidence and cannot be promoted accidentally.
    resonant_ns_seed_windows::Tuple = (
        (zeta=0.25, mode=2, gamma_range=(0.425, 0.445)),
    )
    source_directions::Tuple{Vararg{Symbol}} = (:positive,)
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.99)
    zeta_range::Tuple{Float64,Float64} = (0.001, 0.99)
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    curve_ds::Float64 = 1e-4
    curve_dsmax::Float64 = 5e-4
    curve_max_steps::Int = 2000
    curve_save_sol_every_step::Int = 5
    checkpoint_every::Int = 5
    minimum_component_points::Int = 12
    minimum_component_zeta_span::Float64 = 1e-2
    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
    source_gamma_limits::Tuple{Float64,Float64} = (0.30, 0.99)
    duplicate_tolerance::Float64 = 3e-3
end

"""Resolution presets for fixed-zeta seeded two-parameter continuation."""
function sax_regularized_plane_completion_settings(
        profile::Symbol=:final; kwargs...)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be :smoke, :pilot, or :final"))
    resolution = if profile == :smoke
        (
            source_slices=(),
            resonant_ns_seed_windows=(),
            zeta_range=(0.55, 0.65),
            collocation_intervals=8,
            collocation_degree=2,
            curve_ds=1e-3,
            curve_dsmax=4e-3,
            curve_max_steps=25,
            curve_save_sol_every_step=1,
            checkpoint_every=2,
            newton_tol=1e-8,
        )
    elseif profile == :pilot
        (
            collocation_intervals=25,
            collocation_degree=3,
            curve_ds=5e-4,
            curve_dsmax=2e-3,
            curve_max_steps=500,
            curve_save_sol_every_step=3,
        )
    else
        NamedTuple()
    end
    return _validate_sax_regularized_plane_completion_settings(
        SaxRegularizedPlaneCompletionSettings(;
            merge(resolution, (; kwargs...))...))
end

function _validate_sax_regularized_plane_completion_settings(
        settings::SaxRegularizedPlaneCompletionSettings)
    settings.schema_version == SAX_REGULARIZED_PLANE_COMPLETION_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported plane-completion schema"))
    settings.nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.zeta_range[1] < settings.seed_zeta < settings.zeta_range[2] ||
        throw(ArgumentError("seed_zeta must lie inside zeta_range"))
    settings.collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.collocation_degree >= 2 ||
        throw(ArgumentError("collocation_degree must be at least two"))
    settings.curve_max_steps > 0 ||
        throw(ArgumentError("curve_max_steps must be positive"))
    settings.checkpoint_every > 0 ||
        throw(ArgumentError("checkpoint_every must be positive"))
    settings.minimum_component_points >= 3 || throw(ArgumentError(
        "minimum_component_points must be at least three"))
    settings.minimum_component_zeta_span > 0 || throw(ArgumentError(
        "minimum_component_zeta_span must be positive"))
    settings.source_gamma_limits[1] < settings.source_gamma_limits[2] ||
        throw(ArgumentError("source_gamma_limits must be increasing"))
    all(direction -> direction in (:positive, :negative),
        settings.source_directions) || throw(ArgumentError(
            "source_directions must contain only :positive or :negative"))
    length(unique(settings.source_directions)) ==
        length(settings.source_directions) || throw(ArgumentError(
            "source_directions must be unique"))
    for source in settings.source_slices
        all(name -> hasproperty(source, name),
            (:zeta, :modes, :hopf_gamma_hints)) || throw(ArgumentError(
                "each source slice needs zeta, modes, and hopf_gamma_hints"))
        settings.zeta_range[1] < float(source.zeta) < settings.zeta_range[2] ||
            throw(ArgumentError("source-slice zeta must lie inside zeta_range"))
        !isempty(source.modes) || throw(ArgumentError(
            "source-slice modes cannot be empty"))
        length(source.modes) == length(source.hopf_gamma_hints) ||
            throw(ArgumentError(
                "each source-slice mode needs one Hopf gamma hint"))
        all(mode -> 1 <= mode <= settings.nmodes, source.modes) ||
            throw(ArgumentError("source-slice mode lies outside 1:nmodes"))
        directions = hasproperty(source, :directions) ?
            source.directions : settings.source_directions
        all(direction -> direction in (:positive, :negative), directions) ||
            throw(ArgumentError(
                "source-slice directions must be :positive or :negative"))
        length(unique(directions)) == length(directions) ||
            throw(ArgumentError("source-slice directions must be unique"))
    end
    source_zetas = float.(getproperty.(settings.source_slices, :zeta))
    length(unique(source_zetas)) == length(source_zetas) ||
        throw(ArgumentError("source-slice zeta values must be unique"))
    for window in settings.resonant_ns_seed_windows
        all(name -> hasproperty(window, name),
            (:zeta, :mode, :gamma_range)) || throw(ArgumentError(
                "each resonant NS window needs zeta, mode, and gamma_range"))
        settings.zeta_range[1] < float(window.zeta) < settings.zeta_range[2] ||
            throw(ArgumentError(
                "resonant NS window zeta must lie inside zeta_range"))
        1 <= Int(window.mode) <= settings.nmodes || throw(ArgumentError(
            "resonant NS window mode lies outside 1:nmodes"))
        float(window.gamma_range[1]) < float(window.gamma_range[2]) ||
            throw(ArgumentError(
                "resonant NS window gamma_range must be increasing"))
        any(source -> isapprox(float(source.zeta), float(window.zeta);
                               atol=1e-10, rtol=0),
            settings.source_slices) || throw(ArgumentError(
                "each resonant NS window needs a matching source slice"))
    end
    return settings
end

function _portable_sax_regularized_plane_completion_settings(
        settings::SaxRegularizedPlaneCompletionSettings)
    names = fieldnames(SaxRegularizedPlaneCompletionSettings)
    return NamedTuple{names}(
        Tuple(getfield(settings, name) for name in names))
end

"""Cache layout for independently restartable fixed-zeta seeded curves."""
function sax_regularized_plane_completion_paths(root::AbstractString)
    directory = joinpath(root, "fixed_zeta_plane_completion")
    return (
        directory=directory,
        manifest=joinpath(directory, "manifest.jld2"),
        component=key -> joinpath(directory, "curve_$(key).jld2"),
        source_slices=joinpath(directory, "source_slices"),
    )
end

function _sax_regularized_plane_source_settings(
        source,
        settings::SaxRegularizedPlaneCompletionSettings,
        stability_settings::SaxFixedZetaSchurSettings)
    portable = _portable_sax_fixed_zeta_schur_settings(stability_settings)
    return _validate_sax_fixed_zeta_schur_settings(
        SaxFixedZetaSchurSettings(; merge(portable, (
            zeta=float(source.zeta),
            modes=Tuple(Int.(source.modes)),
            hopf_gamma_hints=Tuple(float.(source.hopf_gamma_hints)),
            directions=hasproperty(source, :directions) ?
                Tuple(Symbol.(source.directions)) : settings.source_directions,
            gamma_range=settings.source_gamma_limits,
            collocation_intervals=settings.collocation_intervals,
            collocation_degree=settings.collocation_degree,
        ))...))
end

function _sax_regularized_plane_source_audits(
        model_p::NamedTuple,
        root::AbstractString,
        settings::SaxRegularizedPlaneCompletionSettings,
        stability_settings::SaxFixedZetaSchurSettings)
    paths = sax_regularized_plane_completion_paths(root)
    primary = load_sax_fixed_zeta_schur_progress(
        model_p, root; settings=stability_settings)
    audits = Any[(
        role=:primary,
        zeta=float(stability_settings.zeta),
        settings=stability_settings,
        progress=primary,
    )]
    for source in settings.source_slices
        source_settings = _sax_regularized_plane_source_settings(
            source, settings, stability_settings)
        progress = load_sax_fixed_zeta_schur_progress(
            model_p, paths.source_slices; settings=source_settings)
        push!(audits, (
            role=:supplemental,
            zeta=float(source.zeta),
            settings=source_settings,
            progress=progress,
        ))
    end
    return audits
end

function _sax_regularized_plane_source_events(source_audits)
    events = Any[]
    for audit in source_audits, event in audit.progress.events
        validation = hasproperty(event, :validation) ? event.validation : nothing
        validation_summary = isnothing(validation) ? nothing : (
            orbit_residual=hasproperty(validation, :orbit_residual) ?
                float(validation.orbit_residual) : NaN,
            methods_agree=hasproperty(validation, :methods_agree) ?
                Bool(validation.methods_agree) : false,
            periodic_schur=hasproperty(validation, :periodic_schur) &&
                    !isnothing(validation.periodic_schur) ? (
                corrected_growth=float(
                    validation.periodic_schur.corrected_growth),
                floquet_angle=float(
                    validation.periodic_schur.floquet_angle),
                classification=Symbol(
                    validation.periodic_schur.classification),
                resonance=Symbol(validation.periodic_schur.resonance),
            ) : nothing,
            floquet_coll=hasproperty(validation, :floquet_coll) &&
                    !isnothing(validation.floquet_coll) ? (
                corrected_growth=float(
                    validation.floquet_coll.corrected_growth),
                floquet_angle=float(
                    validation.floquet_coll.floquet_angle),
                classification=Symbol(
                    validation.floquet_coll.classification),
                resonance=Symbol(validation.floquet_coll.resonance),
            ) : nothing,
        )
        push!(events, (
            type=Symbol(event.type),
            mode=Int(event.mode),
            gamma=float(event.gamma),
            zeta=float(audit.zeta),
            floquet_angle=float(event.floquet_angle),
            status=Symbol(event.status),
            accepted=Bool(event.accepted),
            source=Symbol(event.source),
            audit_role=audit.role,
            validation=validation_summary,
        ))
    end
    sort!(events; by=event -> (
        event.zeta, event.mode, event.gamma, String(event.type)))
    return events
end

function _sax_regularized_plane_bifurcation_settings(
        settings::SaxRegularizedPlaneCompletionSettings)
    return sax_bifurcation_settings(
        :final;
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=settings.zeta_range,
        zeta_seeds=(settings.seed_zeta,),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_curve_ds=settings.curve_ds,
        po_curve_dsmax=settings.curve_dsmax,
        po_curve_max_steps=settings.curve_max_steps,
        po_curve_save_sol_every_step=settings.curve_save_sol_every_step,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_regularized_plane_seed_key(seed)
    gamma_tag = replace(@sprintf("%.9f", seed.gamma), "." => "p")
    zeta_tag = replace(@sprintf("%.6f", seed.zeta), "." => "p")
    return "$(seed.type)_m$(seed.mode)_g$(gamma_tag)_z$(zeta_tag)"
end

function _sax_regularized_plane_seed_duplicate(left, right,
        settings::SaxRegularizedPlaneCompletionSettings)
    return left.type == right.type && left.mode == right.mode &&
        hypot(left.gamma - right.gamma, left.zeta - right.zeta) <=
            settings.duplicate_tolerance
end

function _sax_regularized_plane_seeds(stabilities::AbstractVector,
        settings::SaxRegularizedPlaneCompletionSettings)
    raw = Any[]
    for stability in stabilities
        append!(raw, sax_fixed_zeta_continuation_seeds(
            stability;
            gamma_limits=settings.source_gamma_limits,
            types=(:pd, :ns),
        ))
        for window in settings.resonant_ns_seed_windows
            isapprox(float(stability.settings.zeta), float(window.zeta);
                     atol=1e-10, rtol=0) || continue
            for candidate in sax_fixed_zeta_continuation_seeds(
                    stability;
                    gamma_limits=Tuple(float.(window.gamma_range)),
                    types=(:r2,))
                Int(candidate.mode) == Int(window.mode) || continue
                push!(raw, merge(candidate, (
                    type=:ns,
                    source_type=:r2,
                    continuation_role=:resonant_ns_from_r2,
                )))
            end
        end
    end
    selected = Any[]
    for seed in sort(raw; by=seed -> (
            seed.type == :pd ? 0 : 1,
            seed.mode,
            seed.type == :pd ? abs(seed.zeta - 0.25) : seed.zeta,
            seed.gamma))
        any(stored -> _sax_regularized_plane_seed_duplicate(
            stored, seed, settings), selected) && continue
        push!(selected, seed)
    end
    return [merge(seed, (
        key=_sax_regularized_plane_seed_key(seed),
        checkpoint=merge(seed.checkpoint, (
            key=_sax_regularized_plane_seed_key(seed),
            type=seed.type,
            mode=seed.mode,
            gamma=seed.gamma,
            zeta=seed.zeta,
            floquet_angle=seed.floquet_angle,
            localization_status=hasproperty(seed, :source_type) &&
                    seed.source_type == :r2 ?
                :fixed_zeta_pqz_validated_resonant_ns_seed :
                :fixed_zeta_pqz_validated,
        )),
    )) for seed in selected]
end

_sax_regularized_plane_seeds(stability,
        settings::SaxRegularizedPlaneCompletionSettings) =
    _sax_regularized_plane_seeds(Any[stability], settings)

"""Extract gamma from BifurcationKit's augmented PD or NS inner state."""
function _sax_regularized_plane_gamma(inner, kind::Symbol)
    kind in (:pd, :ns) || throw(ArgumentError(
        "augmented plane kind must be :pd or :ns"))
    if inner isa BK.BorderedArray
        parameter = BK.getp(inner)
        return kind == :ns && parameter isa AbstractVector ?
            float(first(parameter)) : float(parameter)
    end
    # Matrix-based NS continuation flattens [orbit; gamma; Floquet angle].
    # PD appends gamma alone.
    return kind == :ns ? float(inner[end - 1]) : float(inner[end])
end

function _sax_regularized_plane_checkpoint_callback(
        path::AbstractString,
        seed,
        model_p::NamedTuple,
        settings::SaxRegularizedPlaneCompletionSettings,
        direction::Symbol,
        prior_curves,
        prior_codim2,
        verbosity::Integer)
    segments = Vector{Vector{NamedTuple}}([NamedTuple[]])
    last_step = Ref(-1)
    started_ns = time_ns()
    callback = function (z, tangent, step, branch; kwargs...)
        state = get(kwargs, :state, nothing)
        !isnothing(state) && BK.in_bisection(state) && return true
        inner = BK.getvec(z)
        # Matrix-based NS continuation flattens the minimally augmented
        # unknown as [orbit; gamma; Floquet angle]. BK.getp(::Vector) returns
        # its last entry, which is the angle rather than gamma. Reading that
        # value here used to stop every NS curve as soon as theta exceeded the
        # configured gamma maximum. PD has only one appended scalar.
        gamma = _sax_regularized_plane_gamma(inner, seed.type)
        zeta = float(BK.getp(z))
        inside_domain = isfinite(gamma) && isfinite(zeta) &&
            settings.gamma_range[1] <= gamma <= settings.gamma_range[2] &&
            settings.zeta_range[1] <= zeta <= settings.zeta_range[2]
        if !inside_domain
            verbosity > 0 && @info(
                "Regularized fixed-zeta seeded curve reached the requested domain boundary",
                key=seed.key,
                type=seed.type,
                direction,
                gamma,
                zeta,
            )
            return false
        end
        if Int(step) <= last_step[]
            push!(segments, NamedTuple[])
        end
        last_step[] = Int(step)
        push!(segments[end], (gamma=gamma, zeta=zeta, step=Int(step)))
        if verbosity > 0 && (step == 1 || step % settings.checkpoint_every == 0)
            @info(
                "Regularized fixed-zeta seeded curve progress",
                key=seed.key,
                type=seed.type,
                accepted_step=Int(step),
                step_limit=settings.curve_max_steps,
                gamma,
                zeta,
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
        end
        if step == 1 || step % settings.checkpoint_every == 0
            gamma_points = Float64[]
            zeta_points = Float64[]
            for segment in segments
                isempty(segment) && continue
                isempty(gamma_points) || begin
                    push!(gamma_points, NaN)
                    push!(zeta_points, NaN)
                end
                append!(gamma_points, (point.gamma for point in segment))
                append!(zeta_points, (point.zeta for point in segment))
            end
            directional_curve = (
                kind=seed.type,
                gamma=gamma_points,
                zeta=zeta_points,
                mode=seed.mode,
                frequency=NaN,
                source=(
                    analysis=:regularized_fixed_zeta_plane_completion,
                    status=:partial,
                    seed=seed.key,
                    direction=direction,
                    periodic_schur_seed_validated=true,
                ),
                diagnostics=nothing,
            )
            directional_curves = Any[prior_curves..., directional_curve]
            curve = _sax_regularized_plane_combined_curve(
                directional_curves, seed; status=:partial)
            payload = (
                analysis=:regularized_fixed_zeta_plane_component,
                status=:partial,
                key=seed.key,
                seed=seed,
                curves=Any[curve],
                directional_curves=directional_curves,
                periodic_codim2_checkpoints=copy(prior_codim2),
                saved_at_unix=time(),
            )
            _save_sax_transition_mechanism_cache(
                path, :regularized_fixed_zeta_plane_component,
                payload, model_p, settings)
        end
        return true
    end
    return callback
end

function _sax_regularized_plane_direction_settings(
        settings::SaxRegularizedPlaneCompletionSettings,
        direction::Symbol)
    direction in (:positive, :negative) || throw(ArgumentError(
        "direction must be :positive or :negative"))
    names = fieldnames(SaxRegularizedPlaneCompletionSettings)
    portable = NamedTuple{names}(
        Tuple(getfield(settings, name) for name in names))
    signed_ds = direction == :positive ?
        abs(settings.curve_ds) : -abs(settings.curve_ds)
    return SaxRegularizedPlaneCompletionSettings(; merge(portable, (
        curve_ds=signed_ds,
    ))...)
end

function _sax_regularized_plane_seed_covered(
        seed,
        completed_curves,
        settings::SaxRegularizedPlaneCompletionSettings)
    checkpoint = merge(seed.checkpoint, (
        type=seed.type,
        mode=seed.mode,
        gamma=seed.gamma,
        zeta=seed.zeta,
    ))
    return any(curve -> _checkpoint_is_covered_by_curve(
        checkpoint, curve; tolerance=settings.duplicate_tolerance),
        completed_curves)
end

function _sax_regularized_plane_completed_directions(payload)
    hasproperty(payload, :directional_curves) || return (Any[], Set{Symbol}())
    declared = hasproperty(payload, :completed_directions) ?
        Set(Symbol.(payload.completed_directions)) : Set{Symbol}()
    curves = Any[]
    directions = Set{Symbol}()
    for curve in payload.directional_curves
        hasproperty(curve, :source) || continue
        hasproperty(curve.source, :direction) || continue
        direction = Symbol(curve.source.direction)
        complete = direction in declared ||
            (hasproperty(curve.source, :status) &&
             Symbol(curve.source.status) == :complete)
        complete || continue
        push!(curves, curve)
        push!(directions, direction)
    end
    return (curves, directions)
end

function _sax_continue_regularized_plane_seed(
        seed,
        model_p::NamedTuple,
        path::AbstractString,
        settings::SaxRegularizedPlaneCompletionSettings;
        direction::Symbol=:positive,
        prior_curves=Any[],
        prior_codim2=Any[],
        verbosity::Integer=1)
    directional = _sax_regularized_plane_direction_settings(
        settings, direction)
    bifurcation_settings =
        _sax_regularized_plane_bifurcation_settings(directional)
    wrapper, parameters = _sax_periodic_wrapper(
        seed.checkpoint, model_p, bifurcation_settings)
    collocation = BK.get_discretization(wrapper)
    state_dimension, _, _ = size(collocation)
    jacobian = BK.jacobian(
        wrapper, seed.checkpoint.solution, parameters)
    bordered = seed.type == :ns ? Complex.(copy(jacobian)) : copy(jacobian)
    dimension = size(bordered, 1)
    border = sin.(collect(1.0:dimension))
    bordered[end, :] .= border
    bordered[:, end] .= reverse(border)
    bordered[end, end] = 0
    if seed.type == :pd
        bordered[end-state_dimension:end-1, 1:state_dimension] .=
            I(state_dimension)
    else
        bordered[end-state_dimension:end-1,
                 end-state_dimension:end-1] .=
            I(state_dimension) .* cis(seed.floquet_angle)
    end
    rhs = zeros(eltype(bordered), dimension)
    rhs[end] = 1
    right = (bordered \ rhs)[begin:end-1]
    left = (adjoint(bordered) \ rhs)[begin:end-1]
    right ./= norm(right)
    left ./= norm(left)
    options = _sax_periodic_curve_options(bifurcation_settings)
    callback = _sax_regularized_plane_checkpoint_callback(
        path, seed, model_p, settings, direction, prior_curves,
        prior_codim2, verbosity)
    lens_gamma = (BK.@optic _.gamma)
    lens_zeta = (BK.@optic _.zeta)
    if seed.type == :pd
        initial = BK.BorderedArray(
            copy(seed.checkpoint.solution), seed.gamma)
        return BK.continuation_pd(
            wrapper, BK.PALC(), initial, parameters,
            lens_gamma, lens_zeta, left, right, options;
            update_minaug_every_step=1,
            bdlinsolver=BK.MatrixBLS(),
            jacobian_ma=BK.MinAugMatrixBased(),
            # The PD codimension-two event evaluates GPD/CP/R2 directly from
            # the minimally augmented formulation. Asking BifurcationKit for
            # eigen-elements here instead applies the periodic-orbit eigensolver
            # to the augmented vector and causes a dimension mismatch.
            compute_eigen_elements=false,
            usehessian=false,
            bothside=false,
            normC=BK.norminf,
            finalise_solution=callback,
            verbosity=max(0, Int(verbosity) - 1),
            plot=false,
            kind=BK.PDPeriodicOrbitCont(),
        )
    end
    initial = BK.BorderedArray(
        copy(seed.checkpoint.solution),
        [seed.gamma, seed.floquet_angle],
    )
    return BK.continuation_ns(
        wrapper, BK.PALC(), initial, parameters,
        lens_gamma, lens_zeta, left, right, options;
        update_minaug_every_step=1,
        bdlinsolver=BK.MatrixBLS(),
        jacobian_ma=BK.MinAugMatrixBased(),
        compute_eigen_elements=false,
        usehessian=false,
        bothside=false,
        normC=BK.norminf,
        finalise_solution=callback,
        verbosity=max(0, Int(verbosity) - 1),
        plot=false,
        kind=BK.NSPeriodicOrbitCont(),
    )
end


function _sax_regularized_plane_combined_curve(
        curves, seed; status::Symbol=:complete)
    gamma = Float64[]
    zeta = Float64[]
    for curve in curves
        isempty(curve.gamma) && continue
        isempty(gamma) || begin
            push!(gamma, NaN)
            push!(zeta, NaN)
        end
        append!(gamma, float.(curve.gamma))
        append!(zeta, float.(curve.zeta))
    end
    return (
        kind=seed.type,
        gamma=gamma,
        zeta=zeta,
        mode=seed.mode,
        frequency=NaN,
        source=(
            analysis=:regularized_fixed_zeta_plane_completion,
            status=status,
            seed=seed.key,
            periodic_schur_seed_validated=true,
            directions=Tuple(Symbol(curve.source.direction) for curve in curves
                             if hasproperty(curve, :source) &&
                                hasproperty(curve.source, :direction)),
        ),
        diagnostics=nothing,
    )
end

"""Numerical guard against treating a two-point augmented fragment as a curve."""
function _sax_regularized_plane_curve_quality(
        curves,
        settings::SaxRegularizedPlaneCompletionSettings)
    points = Tuple{Float64,Float64}[]
    for curve in curves, index in eachindex(curve.gamma, curve.zeta)
        gamma = float(curve.gamma[index])
        zeta = float(curve.zeta[index])
        isfinite(gamma) && isfinite(zeta) && push!(points, (gamma, zeta))
    end
    unique!(points)
    zeta_span = isempty(points) ? 0.0 :
        maximum(last, points) - minimum(last, points)
    accepted = length(points) >= settings.minimum_component_points &&
        zeta_span >= settings.minimum_component_zeta_span
    return (
        accepted=accepted,
        finite_points=length(points),
        zeta_span=zeta_span,
        minimum_points=settings.minimum_component_points,
        minimum_zeta_span=settings.minimum_component_zeta_span,
    )
end

"""
    compute_sax_regularized_plane_completion(model, root; settings, ...)

Continue every distinct dual-validated PD and NS event found by the primary
and supplemental fixed-`zeta` Periodic-Schur audits. A configured narrow
window may additionally reinterpret a dual-validated near-1:2 complex-pair
root as an NS continuation seed while retaining its `:r2` provenance. Each
source slice and each augmented component has an independent cache, so failure
of one cannot discard the others.
"""
function compute_sax_regularized_plane_completion(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxRegularizedPlaneCompletionSettings=
            SaxRegularizedPlaneCompletionSettings(),
        stability_settings::SaxFixedZetaSchurSettings=
            SaxFixedZetaSchurSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_regularized_plane_completion_settings(settings)
    paths = sax_regularized_plane_completion_paths(root)
    mkpath(paths.directory)
    primary_stability = load_sax_fixed_zeta_schur_progress(
        model_p, root; settings=stability_settings)
    primary_stability.status == :complete || error(
        "fixed-zeta Periodic-Schur audit must be complete before plane continuation")
    failures = Any[]
    source_audits = Any[(
        role=:primary,
        zeta=float(stability_settings.zeta),
        settings=stability_settings,
        progress=primary_stability,
    )]
    for source in settings.source_slices
        source_settings = _sax_regularized_plane_source_settings(
            source, settings, stability_settings)
        verbosity > 0 && @info(
            "Regularized supplemental Periodic-Schur seed audit started",
            zeta=source_settings.zeta,
            modes=source_settings.modes,
            directions=source_settings.directions,
            collocation="$(source_settings.collocation_intervals)x$(source_settings.collocation_degree)",
        )
        source_progress = try
            compute_sax_fixed_zeta_schur(
                model_p,
                paths.source_slices;
                settings=source_settings,
                resume=resume,
                verbosity=verbosity,
            )
        catch err
            err isa InterruptException && rethrow()
            failure = (
                stage=:source_audit,
                zeta=float(source.zeta),
                modes=Tuple(Int.(source.modes)),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            @warn "Supplemental Periodic-Schur seed audit failed; completed slices remain reusable" failure
            load_sax_fixed_zeta_schur_progress(
                model_p, paths.source_slices; settings=source_settings)
        end
        push!(source_audits, (
            role=:supplemental,
            zeta=float(source.zeta),
            settings=source_settings,
            progress=source_progress,
        ))
    end
    available_sources = [audit.progress for audit in source_audits
                         if audit.progress.status in (:complete, :partial)]
    seeds = _sax_regularized_plane_seeds(available_sources, settings)
    isempty(seeds) && error(
        "no dual-validated fixed-zeta PD or NS seed is available")
    completed_curves = Any[]
    for seed in seeds
        path = paths.component(seed.key)
        loaded = resume ? _load_sax_transition_mechanism_cache(
            path, :regularized_fixed_zeta_plane_component,
            model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        if loaded.status == :valid && loaded.payload.status == :complete
            append!(completed_curves, loaded.payload.curves)
            continue
        end
        if _sax_regularized_plane_seed_covered(
                seed, completed_curves, settings)
            payload = (
                analysis=:regularized_fixed_zeta_plane_component,
                status=:complete,
                key=seed.key,
                seed=seed,
                curves=Any[],
                covered_by_existing_component=true,
                saved_at_unix=time(),
            )
            _save_sax_transition_mechanism_cache(
                path, :regularized_fixed_zeta_plane_component,
                payload, model_p, settings)
            verbosity > 0 && @info(
                "Skipping fixed-zeta seed already covered by a completed curve",
                key=seed.key,
                type=seed.type,
                mode=seed.mode,
            )
            continue
        end
        started_ns = time_ns()
        verbosity > 0 && @info(
            "Regularized fixed-zeta seeded curve started",
            key=seed.key,
            type=seed.type,
            mode=seed.mode,
            gamma=seed.gamma,
            zeta=seed.zeta,
        )
        try
            directional_curves, completed_directions =
                loaded.status == :valid ?
                _sax_regularized_plane_completed_directions(loaded.payload) :
                (Any[], Set{Symbol}())
            if completed_directions == Set((:negative, :positive)) &&
                    !_sax_regularized_plane_curve_quality(
                        directional_curves, settings).accepted
                verbosity > 0 && @info(
                    "Restarting an augmented component whose two saved directions form only a fragment",
                    key=seed.key,
                    type=seed.type,
                    mode=seed.mode,
                )
                empty!(directional_curves)
                empty!(completed_directions)
            end
            periodic_codim2_checkpoints = loaded.status == :valid &&
                    hasproperty(loaded.payload,
                                :periodic_codim2_checkpoints) ?
                Any[loaded.payload.periodic_codim2_checkpoints...] : Any[]
            for direction in (:negative, :positive)
                direction in completed_directions && continue
                branch = _sax_continue_regularized_plane_seed(
                    seed, model_p, path, settings;
                    direction=direction,
                    prior_curves=directional_curves,
                    prior_codim2=periodic_codim2_checkpoints,
                    verbosity=verbosity)
                summary = _curve_summary(
                    branch, seed.type;
                    mode=seed.mode,
                    source=(
                        analysis=:regularized_fixed_zeta_plane_completion,
                        status=:complete,
                        seed=seed.key,
                        direction=direction,
                        periodic_schur_seed_validated=true,
                    ),
                    active_parameter=:zeta,
                )
                push!(directional_curves, _portable_sax_curve(summary))
                push!(completed_directions, direction)
                if seed.type == :pd
                    for candidate in _sax_pd_r2_neighbour_checkpoints(
                            branch, seed.checkpoint)
                        _push_unique_periodic_codim2_checkpoint!(
                            periodic_codim2_checkpoints, candidate)
                    end
                end
                partial = (
                    analysis=:regularized_fixed_zeta_plane_component,
                    status=:partial,
                    key=seed.key,
                    seed=seed,
                    curves=Any[_sax_regularized_plane_combined_curve(
                        directional_curves, seed; status=:partial)],
                    directional_curves=copy(directional_curves),
                    periodic_codim2_checkpoints=
                        copy(periodic_codim2_checkpoints),
                    completed_directions=Tuple(sort!(collect(completed_directions))),
                    saved_at_unix=time(),
                )
                _save_sax_transition_mechanism_cache(
                    path, :regularized_fixed_zeta_plane_component,
                    partial, model_p, settings)
            end
            summary = _sax_regularized_plane_combined_curve(
                directional_curves, seed)
            quality = _sax_regularized_plane_curve_quality(
                directional_curves, settings)
            if !quality.accepted
                payload = (
                    analysis=:regularized_fixed_zeta_plane_component,
                    status=:partial,
                    key=seed.key,
                    seed=seed,
                    curves=Any[_sax_regularized_plane_combined_curve(
                        directional_curves, seed; status=:partial)],
                    directional_curves=directional_curves,
                    periodic_codim2_checkpoints=
                        periodic_codim2_checkpoints,
                    completed_directions=
                        Tuple(sort!(collect(completed_directions))),
                    quality=quality,
                    saved_at_unix=time(),
                )
                _save_sax_transition_mechanism_cache(
                    path, :regularized_fixed_zeta_plane_component,
                    payload, model_p, settings)
                failure = (
                    key=seed.key,
                    type=seed.type,
                    mode=seed.mode,
                    exception_type=:IncompleteCurve,
                    error="augmented continuation returned only $(quality.finite_points) finite points over a zeta span of $(quality.zeta_span)",
                )
                push!(failures, failure)
                @warn "Fixed-zeta seeded continuation returned a fragment; it remains partial" failure
                continue
            end
            payload = (
                analysis=:regularized_fixed_zeta_plane_component,
                status=:complete,
                key=seed.key,
                seed=seed,
                curves=Any[summary],
                directional_curves=directional_curves,
                periodic_codim2_checkpoints=periodic_codim2_checkpoints,
                quality=quality,
                saved_at_unix=time(),
            )
            _save_sax_transition_mechanism_cache(
                path, :regularized_fixed_zeta_plane_component,
                payload, model_p, settings)
            push!(completed_curves, first(payload.curves))
            verbosity > 0 && @info(
                "Regularized fixed-zeta seeded curve completed",
                key=seed.key,
                type=seed.type,
                points=length(summary.gamma),
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
        catch err
            err isa InterruptException && rethrow()
            failure = (
                key=seed.key,
                type=seed.type,
                mode=seed.mode,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            @warn "Fixed-zeta seeded periodic bifurcation curve failed; other components and checkpoints remain reusable" failure
        end
    end
    progress = load_sax_regularized_plane_completion_progress(
        model_p, root; settings=settings,
        stability_settings=stability_settings)
    manifest = (
        analysis=:regularized_fixed_zeta_plane_completion,
        status=progress.status,
        counts=progress.counts,
        failures=failures,
        source_audits=[(
            role=audit.role,
            zeta=audit.zeta,
            status=audit.progress.status,
            validated_events=audit.progress.counts.validated_events,
        ) for audit in source_audits],
        source_events=progress.source_events,
        settings=_portable_sax_regularized_plane_completion_settings(settings),
        saved_at_unix=time(),
    )
    _atomic_jld2_save(paths.manifest; manifest)
    return progress
end

"""Load completed and live PD/NS curve components without mutating caches."""
function load_sax_regularized_plane_completion_progress(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxRegularizedPlaneCompletionSettings=
            SaxRegularizedPlaneCompletionSettings(),
        stability_settings::SaxFixedZetaSchurSettings=
            SaxFixedZetaSchurSettings())
    _validate_sax_regularized_plane_completion_settings(settings)
    paths = sax_regularized_plane_completion_paths(root)
    source_audits = _sax_regularized_plane_source_audits(
        model_p, root, settings, stability_settings)
    available_sources = [audit.progress for audit in source_audits
                         if audit.progress.status in (:complete, :partial)]
    seeds = _sax_regularized_plane_seeds(available_sources, settings)
    components = Any[]
    statuses = Any[]
    curves = Any[]
    periodic_codim2_checkpoints = Any[]
    for seed in seeds
        path = paths.component(seed.key)
        loaded = _load_sax_transition_mechanism_cache(
            path, :regularized_fixed_zeta_plane_component,
            model_p, settings)
        status = loaded.status == :valid ? loaded.payload.status : loaded.status
        push!(statuses, (
            key=seed.key,
            type=seed.type,
            mode=seed.mode,
            status=status,
            reason=loaded.reason,
            path=path,
        ))
        loaded.status == :valid || continue
        push!(components, loaded.payload)
        append!(curves, loaded.payload.curves)
        if hasproperty(loaded.payload, :periodic_codim2_checkpoints)
            for candidate in loaded.payload.periodic_codim2_checkpoints
                _push_unique_periodic_codim2_checkpoint!(
                    periodic_codim2_checkpoints, candidate)
            end
        end
    end
    complete = count(component -> component.status == :complete, components)
    partial = count(component -> component.status == :partial, components)
    expected = length(seeds)
    sources_complete = all(audit -> audit.progress.status == :complete,
                           source_audits)
    status = sources_complete && expected > 0 && complete == expected ?
        :complete : isempty(components) &&
        all(audit -> audit.progress.status == :missing, source_audits) ?
            :missing : :partial
    return (
        analysis=:regularized_fixed_zeta_plane_completion,
        status=status,
        settings=_portable_sax_regularized_plane_completion_settings(settings),
        paths=paths,
        source_audits=[(
            role=audit.role,
            zeta=audit.zeta,
            modes=audit.settings.modes,
            status=audit.progress.status,
            counts=audit.progress.counts,
            directory=audit.progress.paths.directory,
        ) for audit in source_audits],
        source_events=_sax_regularized_plane_source_events(source_audits),
        seeds=seeds,
        components=components,
        component_status=statuses,
        curves=curves,
        periodic_codim2_checkpoints=periodic_codim2_checkpoints,
        counts=(
            complete=complete,
            partial=partial,
            expected=expected,
            source_complete=count(
                audit -> audit.progress.status == :complete, source_audits),
            source_expected=length(source_audits),
            pd=count(curve -> curve.kind == :pd, curves),
            ns=count(curve -> curve.kind == :ns, curves),
        ),
    )
end
