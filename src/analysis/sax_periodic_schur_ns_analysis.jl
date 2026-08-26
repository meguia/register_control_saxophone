# Periodic-Schur validation and recovery of the high-gamma NS locus.
#
# The upstream PeriodicSchurBifurcationKit package currently constrains
# BifurcationKit to versions 0.3--0.4.  This project uses BifurcationKit 0.7,
# whose collocation wrapper API has changed.  The small adapter below ports the
# collocation part of FloquetPQZ to the current API while delegating the actual
# generalized periodic Schur decomposition to the registered
# PeriodicSchurDecompositions package.

import PeriodicSchurDecompositions as PSD

const SAX_PERIODIC_SCHUR_NS_SCHEMA_VERSION = 1

"""
Floquet solver using a generalized periodic Schur decomposition (PQZ).

For an autonomous periodic orbit the phase multiplier is exactly one.  A
finite collocation mesh displaces its logarithm slightly from zero, so the
default `phase_normalize=true` subtracts that numerical drift from the full
spectrum before BifurcationKit classifies a crossing.
"""
struct SaxFloquetPQZ <: BK.AbstractFloquetSolver
    phase_normalize::Bool
    cyclic_retries::Int
    fallback_to_floquet_coll::Bool
end

SaxFloquetPQZ(phase_normalize::Bool) =
    SaxFloquetPQZ(phase_normalize, 4, true)

SaxFloquetPQZ(;
        phase_normalize::Bool=true,
        cyclic_retries::Integer=4,
        fallback_to_floquet_coll::Bool=true) =
    SaxFloquetPQZ(
        phase_normalize,
        max(0, Int(cyclic_retries)),
        fallback_to_floquet_coll,
    )

BK.geteigenvector(::SaxFloquetPQZ, eigenvectors, indices) = nothing

"""
    sax_collocation_pqz_factors(collocation, jacobian)

Condense every collocation interval independently into the matrix pairs
`(A_j,B_j)` defining the generalized periodic product.  Unlike the historical
plugin implementation, this routine never forms a dense transformation with
the size of the complete collocation Jacobian.
"""
function sax_collocation_pqz_factors(collocation::BK.Collocation,
                                     jacobian::AbstractMatrix)
    state_dimension, degree, intervals = size(collocation)
    block_dimension = state_dimension * degree
    expected = state_dimension * (1 + degree * intervals) + 1
    size(jacobian) == (expected, expected) || throw(DimensionMismatch(
        "collocation Jacobian has size $(size(jacobian)); expected " *
        "($(expected), $(expected))",
    ))

    As = Matrix{eltype(jacobian)}[]
    Bs = Matrix{eltype(jacobian)}[]
    equation_rows = 1:block_dimension
    left_columns = 1:state_dimension
    right_columns = left_columns .+ block_dimension

    for _ in 1:intervals
        internal = lu(Matrix(@view jacobian[equation_rows,
                                            equation_rows .+ state_dimension]))
        # F.L * F.U == F.P * block.  Applying inv(F.L) * F.P to
        # the two boundary blocks is precisely the local condensation used by
        # FloquetPQZ before the periodic generalized Schur decomposition.
        left = internal.L \ Matrix(@view jacobian[equation_rows,
                                                   left_columns])[internal.p, :]
        right = internal.L \ Matrix(@view jacobian[equation_rows,
                                                    right_columns])[internal.p, :]
        tail = (block_dimension - state_dimension + 1):block_dimension
        push!(As, Matrix(@view left[tail, :]))
        push!(Bs, -Matrix(@view right[tail, :]))

        equation_rows = equation_rows .+ block_dimension
        left_columns = left_columns .+ block_dimension
        right_columns = right_columns .+ block_dimension
    end
    return (As=As, Bs=Bs)
end

"""Compute the leading Floquet exponents with generalized periodic Schur."""
function sax_periodic_schur_exponents(collocation::BK.Collocation,
                                      jacobian::AbstractMatrix,
                                      nev::Integer;
                                      phase_normalize::Bool=true,
                                      cyclic_retries::Integer=4)
    factors = sax_collocation_pqz_factors(collocation, jacobian)
    factor_count = length(factors.As)
    attempts = min(factor_count, 1 + max(0, Int(cyclic_retries)))
    decomposition = nothing
    last_error = nothing
    for offset in 0:(attempts - 1)
        As = offset == 0 ? factors.As : circshift(factors.As, -offset)
        Bs = offset == 0 ? factors.Bs : circshift(factors.Bs, -offset)
        try
            decomposition = PSD.gpschur(
                As,
                Bs;
                wantZ=false,
                wantT=false,
            )
            break
        catch err
            err isa InterruptException && rethrow()
            last_error = err
        end
    end
    isnothing(decomposition) && throw(last_error)
    exponents = log.(complex.(decomposition.values))
    if phase_normalize
        neutral = exponents[argmin(abs.(exponents))]
        exponents .-= neutral
    end
    count = min(Int(nev), length(exponents))
    ordering = sortperm(exponents; by=real, rev=true)[1:count]
    return ComplexF64.(exponents[ordering]), nothing, true, 1
end

function (solver::SaxFloquetPQZ)(collocation::BK.Collocation,
                                 jacobian::AbstractMatrix,
                                 nev::Integer;
                                 kwargs...)
    try
        return sax_periodic_schur_exponents(
            collocation, jacobian, nev;
            phase_normalize=solver.phase_normalize,
            cyclic_retries=solver.cyclic_retries,
        )
    catch err
        err isa InterruptException && rethrow()
        solver.fallback_to_floquet_coll || rethrow()
        @warn(
            "PQZ scan step failed after cyclic retries; using FloquetColl for this continuation sample",
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
            maxlog=5,
        )
        values, vectors, converged, iterations = BK.FloquetColl()(
            collocation, jacobian, nev)
        normalized = ComplexF64.(values)
        if solver.phase_normalize && !isempty(normalized)
            neutral = normalized[argmin(abs.(normalized))]
            normalized .-= neutral
        end
        return normalized, vectors, converged, iterations
    end
end

function BK.compute_eigenvalues(solver::SaxFloquetPQZ,
                                iterator::BK.ContIterable{Tkind},
                                state,
                                orbit,
                                parameters,
                                nev=iterator.contparams.nev;
                                kwargs...) where {Tkind <: BK.AbstractContinuationKind}
    wrapper = BK.get_wrap_po(iterator)
    collocation = BK.get_discretization(wrapper)
    jacobian = BK.jacobian(wrapper, orbit, parameters)
    return solver(collocation, jacobian, nev; kwargs...)
end

# ---------------------------------------------------------------------------
# Settings and spectral diagnostics
# ---------------------------------------------------------------------------

"""Numerical controls for Periodic-Schur NS validation and recovery."""
Base.@kwdef struct SaxPeriodicSchurNSSettings
    schema_version::Int = SAX_PERIODIC_SCHUR_NS_SCHEMA_VERSION
    nmodes::Int = 8
    mode::Int = 2
    gamma_hint::Float64 = 0.6043477221884095
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.72)
    root_gamma_range::Tuple{Float64,Float64} = (0.52, 0.72)
    zeta_range::Tuple{Float64,Float64} = (0.10, 0.90)
    zeta_values::Tuple{Vararg{Float64}} =
        Tuple(collect(range(0.10, 0.90; step=0.025)))
    validation_meshes::Tuple{Vararg{Tuple{Int,Int}}} =
        ((25, 3), (40, 4), (60, 5))
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    po_ds::Float64 = 5e-4
    po_dsmax::Float64 = 3e-3
    po_max_steps::Int = 650
    hopf_scan_points::Int = 241
    newton_tol::Float64 = 1e-10
    newton_max_iterations::Int = 45
    stability_tol::Float64 = 1e-8
    root_growth_tolerance::Float64 = 2e-5
    method_growth_tolerance::Float64 = 2e-7
    method_angle_tolerance::Float64 = 2e-7
    minimum_ns_angle::Float64 = 1e-3
    minimum_angle_to_pi::Float64 = 1e-3
    r2_warning_angle::Float64 = 0.08
    augmented_ds::Float64 = 2e-5
    augmented_dsmin::Float64 = 1e-11
    augmented_dsmax::Float64 = 1e-4
    augmented_max_steps::Int = 1200
    augmented_checkpoint_every::Int = 5
    augmented_progress_every::Int = 5
    augmented_formulations::Tuple{Vararg{Symbol}} =
        (:minaug, :matrix_based)
end

function sax_periodic_schur_ns_settings(profile::Symbol=:final)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "Periodic-Schur NS profile must be :smoke, :pilot, or :final",
    ))
    if profile == :smoke
        return SaxPeriodicSchurNSSettings(
            gamma_range=(0.30, 0.64),
            root_gamma_range=(0.58, 0.63),
            zeta_range=(0.25, 0.25),
            zeta_values=(0.25,),
            validation_meshes=((25, 3), (40, 4)),
            collocation_intervals=25,
            collocation_degree=3,
            po_ds=2e-3,
            po_dsmax=5e-3,
            po_max_steps=120,
            hopf_scan_points=101,
            augmented_max_steps=2,
            augmented_checkpoint_every=1,
            augmented_progress_every=1,
            augmented_formulations=(:matrix_based,),
        )
    elseif profile == :pilot
        return SaxPeriodicSchurNSSettings(
            zeta_range=(0.15, 0.45),
            zeta_values=Tuple(collect(range(0.15, 0.45; step=0.05))),
            validation_meshes=((25, 3), (40, 4)),
            collocation_intervals=25,
            collocation_degree=3,
            po_ds=1e-3,
            po_dsmax=5e-3,
            po_max_steps=350,
            hopf_scan_points=161,
            augmented_max_steps=250,
        )
    end
    return SaxPeriodicSchurNSSettings()
end

function _validate_sax_periodic_schur_ns_settings(
        settings::SaxPeriodicSchurNSSettings)
    settings.schema_version == SAX_PERIODIC_SCHUR_NS_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported Periodic-Schur NS schema version"))
    1 <= settings.mode <= settings.nmodes || throw(ArgumentError(
        "Periodic-Schur NS mode must lie in 1:nmodes",
    ))
    settings.gamma_range[1] < settings.gamma_range[2] || throw(ArgumentError(
        "Periodic-Schur gamma_range must be increasing",
    ))
    settings.root_gamma_range[1] < settings.root_gamma_range[2] ||
        throw(ArgumentError("Periodic-Schur root_gamma_range must be increasing"))
    settings.gamma_range[1] <= settings.root_gamma_range[1] &&
        settings.root_gamma_range[2] <= settings.gamma_range[2] ||
        throw(ArgumentError("root_gamma_range must lie inside gamma_range"))
    !isempty(settings.zeta_values) || throw(ArgumentError(
        "Periodic-Schur zeta_values cannot be empty",
    ))
    all(z -> settings.zeta_range[1] <= z <= settings.zeta_range[2],
        settings.zeta_values) || throw(ArgumentError(
        "all Periodic-Schur zeta values must lie inside zeta_range",
    ))
    all(mesh -> mesh[1] >= 5 && mesh[2] >= 2,
        settings.validation_meshes) || throw(ArgumentError(
        "Periodic-Schur validation meshes are too small",
    ))
    0 < settings.minimum_ns_angle < pi || throw(ArgumentError(
        "minimum_ns_angle must lie in (0, pi)",
    ))
    0 < settings.minimum_angle_to_pi < pi || throw(ArgumentError(
        "minimum_angle_to_pi must lie in (0, pi)",
    ))
    all(formulation -> formulation in (:minaug, :matrix_based),
        settings.augmented_formulations) || throw(ArgumentError(
        "augmented formulations must be :minaug or :matrix_based",
    ))
    return settings
end

function _sax_floquet_spectrum_summary(exponents,
                                       target_angle::Real,
                                       settings::SaxPeriodicSchurNSSettings;
                                       method::Symbol)
    values = ComplexF64.(exponents)
    neutral_index = argmin(abs.(values))
    neutral = values[neutral_index]
    available = [index for index in eachindex(values) if index != neutral_index]
    isempty(available) && error("Floquet spectrum contains only the trivial exponent")
    target = abs(float(target_angle))
    critical_index = available[argmin(hypot(
        real(values[index]) - real(neutral),
        abs(abs(imag(values[index])) - target),
    ) for index in available)]
    critical = values[critical_index]
    corrected = critical - neutral
    angle = abs(imag(corrected))
    angle_to_pi = abs(pi - angle)
    other = [index for index in eachindex(values)
             if index != neutral_index && index != critical_index &&
                abs(values[index] - conj(critical)) > 1e-8]
    spectral_gap = isempty(other) ? Inf :
        minimum(abs(values[index] - critical) for index in other)
    valid = abs(real(corrected)) <= settings.root_growth_tolerance &&
            angle >= settings.minimum_ns_angle &&
            angle_to_pi >= settings.minimum_angle_to_pi
    return (
        method=method,
        exponents=values,
        neutral_exponent=neutral,
        critical_exponent=critical,
        corrected_exponent=corrected,
        corrected_growth=float(real(corrected)),
        floquet_angle=float(angle),
        angle_to_pi=float(angle_to_pi),
        multiplier=ComplexF64(exp(corrected)),
        trivial_error=float(abs(neutral)),
        spectral_gap=float(spectral_gap),
        near_r2=angle_to_pi <= settings.r2_warning_angle,
        valid=valid,
    )
end

function _sax_dual_floquet_spectrum(checkpoint,
                                    model_p::NamedTuple,
                                    bifurcation_settings::SaxBifurcationSettings,
                                    settings::SaxPeriodicSchurNSSettings)
    wrapper, parameters = _sax_periodic_wrapper(
        checkpoint, model_p, bifurcation_settings)
    collocation = BK.get_discretization(wrapper)
    jacobian = BK.jacobian(wrapper, checkpoint.solution, parameters)
    nev = 2 + 2settings.nmodes
    coll_values, _, coll_converged, coll_iterations = BK.FloquetColl()(
        collocation, jacobian, nev)
    pqz_values, _, pqz_converged, pqz_iterations = SaxFloquetPQZ(
        fallback_to_floquet_coll=false)(
        collocation, jacobian, nev)
    coll = merge(_sax_floquet_spectrum_summary(
        coll_values, checkpoint.floquet_angle, settings; method=:floquet_coll),
        (converged=Bool(coll_converged), iterations=Int(coll_iterations)),
    )
    pqz = merge(_sax_floquet_spectrum_summary(
        pqz_values, checkpoint.floquet_angle, settings; method=:periodic_schur),
        (converged=Bool(pqz_converged), iterations=Int(pqz_iterations)),
    )
    growth_difference = abs(coll.corrected_growth - pqz.corrected_growth)
    angle_difference = abs(coll.floquet_angle - pqz.floquet_angle)
    return (
        gamma=float(checkpoint.gamma),
        zeta=float(checkpoint.zeta),
        mesh=(bifurcation_settings.po_collocation_intervals,
              bifurcation_settings.po_collocation_degree),
        orbit_residual=float(norm(
            BK.residual(wrapper, checkpoint.solution, parameters), Inf)),
        floquet_coll=coll,
        periodic_schur=pqz,
        growth_difference=float(growth_difference),
        angle_difference=float(angle_difference),
        methods_agree=growth_difference <= settings.method_growth_tolerance &&
                      angle_difference <= settings.method_angle_tolerance,
        classification_agrees=coll.valid == pqz.valid &&
                              coll.near_r2 == pqz.near_r2,
    )
end

# ---------------------------------------------------------------------------
# Mesh-convergence validation of the ordinary high-gamma candidate
# ---------------------------------------------------------------------------

function _sax_regrid_periodic_checkpoint(checkpoint,
                                         model_p::NamedTuple,
                                         intervals::Integer,
                                         degree::Integer,
                                         settings::SaxPeriodicSchurNSSettings)
    source_settings = _sax_mechanism_bifurcation_settings(
        settings.nmodes, 40, 4;
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    source_wrapper, source_parameters = _sax_periodic_wrapper(
        checkpoint, model_p, source_settings)
    source_collocation = BK.get_discretization(source_wrapper)
    interpolant = BK.POSolution(source_collocation, checkpoint.solution)

    target_settings = _sax_mechanism_bifurcation_settings(
        settings.nmodes, intervals, degree;
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    state_dimension = 2 + 2settings.nmodes
    problem_checkpoint = (
        state=collect(checkpoint.solution[1:state_dimension]),
        gamma=checkpoint.gamma,
        zeta=checkpoint.zeta,
    )
    problem, parameters = _sax_problem_from_checkpoint(
        problem_checkpoint, model_p, target_settings)
    orbit_dimension = state_dimension * (1 + Int(intervals) * Int(degree))
    collocation = BK.Collocation(
        Int(intervals),
        Int(degree);
        N=state_dimension,
        prob_vf=problem,
        ϕ=zeros(orbit_dimension),
        xπ=zeros(orbit_dimension),
        ∂ϕ=zeros(state_dimension, Int(intervals) * Int(degree)),
        jacobian=BK.DenseAnalyticalInplace(),
        update_section_every_step=1,
    )
    guess = BK.generate_solution(
        collocation, interpolant, float(last(checkpoint.solution)))
    BK.updatesection!(collocation, guess, parameters)
    corrected = BK.newton(
        collocation,
        guess,
        BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=settings.newton_max_iterations,
            linsolver=BK.COPLS(),
            verbose=false,
        );
        normN=BK.norminf,
    )
    BK.converged(corrected) || error(
        "periodic-orbit correction failed on mesh $(intervals) x $(degree)")
    refined = merge(checkpoint, (
        key="$(checkpoint.key)_mesh$(Int(intervals))x$(Int(degree))",
        solution=collect(float.(corrected.u)),
    ))
    return refined, target_settings
end

"""Validate the ordinary high-gamma candidate on several meshes and solvers."""
function compute_sax_periodic_schur_seed_validation(
        model_p::NamedTuple,
        stage_directory::AbstractString,
        output_path::AbstractString;
        settings::SaxPeriodicSchurNSSettings=SaxPeriodicSchurNSSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_periodic_schur_ns_settings(settings)
    cached = resume ? _load_sax_transition_mechanism_cache(
        output_path, :periodic_schur_seed_validation, model_p, settings) :
        (status=:missing, payload=nothing, reason="resume disabled")
    cached.status == :valid && return cached.payload
    source = load_sax_periodic_stage_checkpoint(
        stage_directory,
        model_p;
        kind=:ns,
        mode=settings.mode,
        gamma_hint=settings.gamma_hint,
        settings=sax_bifurcation_settings(:final),
    )
    source.status == :valid || error(
        "ordinary high-gamma NS candidate is unavailable: $(source.reason)")
    rows = Any[]
    for mesh in settings.validation_meshes
        verbosity > 0 && @info(
            "Periodic-Schur seed mesh validation", intervals=mesh[1], degree=mesh[2])
        refined, bif_settings = _sax_regrid_periodic_checkpoint(
            source.checkpoint, model_p, mesh[1], mesh[2], settings)
        push!(rows, merge(
            _sax_dual_floquet_spectrum(
                refined, model_p, bif_settings, settings),
            (checkpoint=refined,),
        ))
    end
    result = (
        analysis=:periodic_schur_seed_validation,
        source_path=abspath(source.path),
        source_checkpoint=source.checkpoint,
        rows=rows,
        methods_agree=all(row.methods_agree for row in rows),
        mesh_converged=length(rows) < 2 || (
            abs(rows[end].periodic_schur.corrected_growth -
                rows[end - 1].periodic_schur.corrected_growth) <=
                settings.root_growth_tolerance &&
            abs(rows[end].periodic_schur.floquet_angle -
                rows[end - 1].periodic_schur.floquet_angle) <=
                settings.method_angle_tolerance
        ),
        candidate_valid=rows[end].periodic_schur.valid,
        settings=_portable_sax_mechanism_settings(settings),
    )
    return _save_sax_transition_mechanism_cache(
        output_path, :periodic_schur_seed_validation,
        result, model_p, settings)
end

# ---------------------------------------------------------------------------
# Independent fixed-zeta PQZ root slices
# ---------------------------------------------------------------------------

function _sax_periodic_schur_slice_tag(zeta::Real)
    return replace(@sprintf("%.6f", float(zeta)), "." => "p", "-" => "m")
end

function _sax_periodic_schur_slice_path(directory::AbstractString, zeta::Real)
    return joinpath(directory,
                    "pqz_ns_slice_z$(_sax_periodic_schur_slice_tag(zeta)).jld2")
end

function _sax_periodic_schur_bifurcation_settings(
        settings::SaxPeriodicSchurNSSettings)
    return _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        settings.collocation_intervals,
        settings.collocation_degree;
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=0,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_periodic_schur_rescue_settings(
        settings::SaxPeriodicSchurNSSettings,
        zeta::Real)
    return SaxPDRescueSettings(
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        seed_zetas=(float(zeta),),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=0,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
        hopf_scan_points=settings.hopf_scan_points,
    )
end

function _sax_periodic_schur_branch_samples(branch, zeta::Real)
    count = min(length(branch), length(branch.eig))
    samples = Any[]
    for index in 1:count
        eigen_record = branch.eig[index]
        hasproperty(eigen_record, :eigenvals) || continue
        values = ComplexF64.(eigen_record.eigenvals)
        isempty(values) && continue
        neutral_index = argmin(abs.(values))
        neutral = values[neutral_index]
        available = [i for i in eachindex(values) if i != neutral_index]
        isempty(available) && continue
        dominant_index = available[argmax(real(values[i] - neutral)
                                           for i in available)]
        dominant = values[dominant_index] - neutral
        push!(samples, (
            branch_index=Int(index),
            gamma=float(branch.branch.gamma[index]),
            zeta=float(zeta),
            period=float(branch.branch.period[index]),
            neutral_exponent=neutral,
            dominant_exponent=dominant,
            dominant_growth=float(real(dominant)),
            dominant_angle=float(abs(imag(dominant))),
            dominant_angle_to_pi=float(abs(pi - abs(imag(dominant)))),
        ))
    end
    return samples
end

"""
    compute_sax_periodic_schur_ns_slice(model_p, zeta; settings, verbosity=1)

Continue the mode-2 period-one family at fixed `zeta` using `SaxFloquetPQZ`
as BifurcationKit's stability solver.  Every independently localized complex
unit-circle crossing is then recomputed with both PQZ and `FloquetColl`.
"""
function compute_sax_periodic_schur_ns_slice(
        model_p::NamedTuple,
        zeta::Real;
        settings::SaxPeriodicSchurNSSettings=SaxPeriodicSchurNSSettings(),
        verbosity::Integer=1)
    _validate_sax_periodic_schur_ns_settings(settings)
    rescue_settings = _sax_periodic_schur_rescue_settings(settings, zeta)
    hopf = refine_sax_hopf_checkpoint(
        model_p,
        zeta,
        settings.mode;
        settings=rescue_settings,
    )
    bifurcation_settings = _sax_periodic_schur_bifurcation_settings(settings)
    branch = continue_sax_periodic_orbits(
        hopf,
        model_p;
        settings=bifurcation_settings,
        verbosity=Int(verbosity),
        eigsolver=SaxFloquetPQZ(),
    )
    checkpoints = _sax_periodic_bifurcation_checkpoints(
        branch, hopf, settings.mode)
    ns_checkpoints = [checkpoint for checkpoint in checkpoints
                      if checkpoint.type == :ns &&
                         settings.root_gamma_range[1] <= checkpoint.gamma <=
                         settings.root_gamma_range[2]]
    roots = map(ns_checkpoints) do checkpoint
        validation = _sax_dual_floquet_spectrum(
            checkpoint, model_p, bifurcation_settings, settings)
        (
            checkpoint=checkpoint,
            validation=validation,
            accepted=validation.periodic_schur.valid &&
                     validation.floquet_coll.valid &&
                     validation.methods_agree &&
                     validation.classification_agrees &&
                     validation.orbit_residual <= 100 * settings.newton_tol,
            near_r2=validation.periodic_schur.near_r2,
        )
    end
    return (
        analysis=:periodic_schur_ns_slice,
        zeta=float(zeta),
        hopf_checkpoint=hopf,
        samples=_sax_periodic_schur_branch_samples(branch, zeta),
        bifurcations=[(
            type=checkpoint.type,
            gamma=checkpoint.gamma,
            zeta=checkpoint.zeta,
            floquet_angle=checkpoint.floquet_angle,
            localization_status=checkpoint.localization_status,
            localization_precision=checkpoint.localization_precision,
        ) for checkpoint in checkpoints],
        roots=roots,
        accepted_roots=count(root -> root.accepted, roots),
        periodic_diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
        settings=_portable_sax_mechanism_settings(settings),
    )
end

"""Compute or resume all fixed-zeta PQZ slices with one atomic file per row."""
function compute_sax_periodic_schur_ns_slices(
        model_p::NamedTuple,
        output_directory::AbstractString;
        settings::SaxPeriodicSchurNSSettings=SaxPeriodicSchurNSSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_periodic_schur_ns_settings(settings)
    mkpath(output_directory)
    slices = Any[]
    failures = Any[]
    for (index, zeta) in enumerate(settings.zeta_values)
        path = _sax_periodic_schur_slice_path(output_directory, zeta)
        cached = resume ? _load_sax_transition_mechanism_cache(
            path, :periodic_schur_ns_slice, model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        if cached.status == :valid
            push!(slices, cached.payload)
            continue
        end
        verbosity > 0 && @info(
            "Starting Periodic-Schur NS slice",
            slice=index,
            total=length(settings.zeta_values),
            zeta=float(zeta),
        )
        try
            slice = compute_sax_periodic_schur_ns_slice(
                model_p, zeta; settings=settings, verbosity=verbosity)
            _save_sax_transition_mechanism_cache(
                path, :periodic_schur_ns_slice, slice, model_p, settings)
            push!(slices, slice)
        catch err
            err isa InterruptException && rethrow()
            failure = (
                zeta=float(zeta),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            verbosity > 0 && @warn "Periodic-Schur NS slice failed" failure
        end
    end
    ordered = sort(slices; by=slice -> slice.zeta)
    result = (
        analysis=:periodic_schur_ns_slices,
        slices=ordered,
        roots=Any[root for slice in ordered for root in slice.roots],
        failures=failures,
        completed_slices=length(ordered),
        expected_slices=length(settings.zeta_values),
        settings=_portable_sax_mechanism_settings(settings),
    )
    return _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "pqz_ns_slice_manifest.jld2"),
        :periodic_schur_ns_slices, result, model_p, settings)
end

function _sax_periodic_schur_root_score(root)
    validation = root.validation.periodic_schur
    return (
        root.accepted ? 0 : 1,
        validation.near_r2 ? 1 : 0,
        abs(validation.corrected_growth),
        -validation.spectral_gap,
        -validation.angle_to_pi,
    )
end

"""Choose and cache the best independently localized PQZ NS root."""
function select_sax_periodic_schur_ns_seed(
        model_p::NamedTuple,
        slices_result,
        output_path::AbstractString;
        settings::SaxPeriodicSchurNSSettings=SaxPeriodicSchurNSSettings())
    roots = [root for root in slices_result.roots if root.accepted]
    result = if isempty(roots)
        (
            analysis=:periodic_schur_ns_seed,
            status=:missing,
            seed=nothing,
            reason="no independently localized PQZ root passed validation",
        )
    else
        selected = roots[argmin(_sax_periodic_schur_root_score(root)
                                for root in roots)]
        seed = merge(selected.checkpoint, (
            key="pqz_refined_$(selected.checkpoint.key)",
            localization_status=:pqz_dual_validated,
            localization_precision=abs(
                selected.validation.periodic_schur.corrected_growth),
        ))
        (
            analysis=:periodic_schur_ns_seed,
            status=:valid,
            seed=seed,
            validation=selected.validation,
            reason="best dual-validated PQZ slice root",
        )
    end
    return _save_sax_transition_mechanism_cache(
        output_path, :periodic_schur_ns_seed, result, model_p, settings)
end

# ---------------------------------------------------------------------------
# Augmented continuation from the refined PQZ root
# ---------------------------------------------------------------------------

function _sax_periodic_schur_augmented_settings(
        settings::SaxPeriodicSchurNSSettings)
    return SaxHighGammaNSSettings(
        nmodes=settings.nmodes,
        mode=settings.mode,
        seed_zeta=first(settings.zeta_values),
        gamma_hint=settings.gamma_hint,
        zeta_range=settings.zeta_range,
        collocation_intervals=settings.collocation_intervals,
        collocation_degree=settings.collocation_degree,
        ds=settings.augmented_ds,
        dsmin=settings.augmented_dsmin,
        dsmax=settings.augmented_dsmax,
        max_steps=settings.augmented_max_steps,
        save_sol_every_step=settings.augmented_checkpoint_every,
        checkpoint_every=settings.augmented_checkpoint_every,
        progress_every=settings.augmented_progress_every,
        codim2_detection=2,
        newton_tol=settings.newton_tol,
        newton_max_iterations=settings.newton_max_iterations,
        stability_tol=settings.stability_tol,
        validation_residual_tol=settings.newton_tol * 100,
        validation_floquet_tol=settings.root_growth_tolerance,
        r2_angle_warning=settings.r2_warning_angle,
        formulations=settings.augmented_formulations,
    )
end

function _sax_periodic_schur_augmented_paths(directory::AbstractString,
                                             formulation::Symbol,
                                             direction::Symbol)
    stem = "pqz_augmented_ns_$(formulation)_$(direction)"
    return (
        result=joinpath(directory, "$(stem).jld2"),
        checkpoints=joinpath(directory, "$(stem)_checkpoints.jld2"),
    )
end

"""Continue the refined PQZ seed in both zeta directions with checkpoints."""
function compute_sax_periodic_schur_augmented_ns(
        model_p::NamedTuple,
        seed_result,
        output_directory::AbstractString;
        settings::SaxPeriodicSchurNSSettings=SaxPeriodicSchurNSSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    mkpath(output_directory)
    if seed_result.status != :valid
        result = (
            analysis=:periodic_schur_augmented_ns,
            status=:no_seed,
            seed=nothing,
            components=Any[],
            failures=Any[],
            reason=seed_result.reason,
        )
        return _save_sax_transition_mechanism_cache(
            joinpath(output_directory, "pqz_augmented_ns_manifest.jld2"),
            :periodic_schur_augmented_ns, result, model_p, settings)
    end
    augmented_settings = _sax_periodic_schur_augmented_settings(settings)
    seed = seed_result.seed
    components = Any[]
    failures = Any[]
    for direction in (:positive, :negative)
        accepted = false
        for formulation in settings.augmented_formulations
            accepted && break
            paths = _sax_periodic_schur_augmented_paths(
                output_directory, formulation, direction)
            cached = resume ? _load_sax_transition_mechanism_cache(
                paths.result, :periodic_schur_augmented_component,
                model_p, settings) :
                (status=:missing, payload=nothing, reason="resume disabled")
            component = if cached.status == :valid
                cached.payload
            else
                verbosity > 0 && @info(
                    "Starting refined PQZ augmented NS continuation",
                    formulation,
                    direction,
                    gamma=seed.gamma,
                    zeta=seed.zeta,
                    theta=seed.floquet_angle,
                )
                run = continue_sax_high_gamma_ns(
                    seed,
                    model_p;
                    settings=augmented_settings,
                    formulation=formulation,
                    direction=direction,
                    checkpoint_path=paths.checkpoints,
                    verbosity=verbosity,
                )
                _save_sax_transition_mechanism_cache(
                    paths.result, :periodic_schur_augmented_component,
                    run, model_p, settings)
            end
            push!(components, component)
            accepted = component.checkpoint_count > 0 ||
                       !isempty(component.curve.gamma)
            if !accepted
                push!(failures, (
                    formulation=formulation,
                    direction=direction,
                    status=component.status,
                    error=component.terminal_error,
                ))
            end
        end
    end
    result = (
        analysis=:periodic_schur_augmented_ns,
        status=any(component -> !isempty(component.curve.gamma), components) ?
            :partial_or_complete : :failed,
        seed=seed,
        components=components,
        failures=failures,
        reason=isempty(failures) ? "both directional continuations accepted" :
            "one or more augmented continuations failed",
    )
    return _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "pqz_augmented_ns_manifest.jld2"),
        :periodic_schur_augmented_ns, result, model_p, settings)
end

# ---------------------------------------------------------------------------
# Read-only progress loading and notebook visualization
# ---------------------------------------------------------------------------

function _sax_periodic_schur_root_locus(slices)
    points = Any[]
    for slice in slices
        accepted = [root for root in slice.roots if root.accepted]
        isempty(accepted) && continue
        root = accepted[argmin(_sax_periodic_schur_root_score(candidate)
                               for candidate in accepted)]
        push!(points, (
            gamma=float(root.checkpoint.gamma),
            zeta=float(root.checkpoint.zeta),
            corrected_growth=float(
                root.validation.periodic_schur.corrected_growth),
            floquet_angle=float(root.validation.periodic_schur.floquet_angle),
            growth_difference=float(root.validation.growth_difference),
            angle_difference=float(root.validation.angle_difference),
        ))
    end
    sort!(points; by=point -> point.zeta)
    return (
        kind=:ns,
        mode=2,
        gamma=[point.gamma for point in points],
        zeta=[point.zeta for point in points],
        points=points,
        source=(analysis=:periodic_schur_fixed_zeta_roots,
                provisional=true, independent_slices=true),
    )
end

function _sax_periodic_schur_bracket_angle(slice, gamma::Real)
    samples = [sample for sample in slice.samples
               if isfinite(sample.gamma) && isfinite(sample.dominant_angle)]
    isempty(samples) && return NaN
    nearest = samples[argmin(abs(sample.gamma - gamma) for sample in samples)]
    return float(nearest.dominant_angle)
end

"""
    sax_periodic_schur_provisional_brackets(slices, settings)

Extract unresolved Floquet sign-change brackets retained by BifurcationKit as
`status=:guess`. Overlapping brackets from the two continuation directions are
merged. The result is diagnostic only: it records where event localization
failed and must not be interpreted as a validated NS or PD curve.
"""
function sax_periodic_schur_provisional_brackets(
        slices,
        settings::SaxPeriodicSchurNSSettings;
        merge_gap::Real=5e-4)
    brackets = Any[]
    for slice in slices
        raw = Any[]
        specialpoints = try
            slice.periodic_diagnostics.specialpoints
        catch
            Any[]
        end
        for special in specialpoints
            special.type in (:ns, :pd) || continue
            special.status == :guess || continue
            interval = Tuple(float.(special.interval))
            lower, upper = extrema(interval)
            lower = max(lower, settings.root_gamma_range[1])
            upper = min(upper, settings.root_gamma_range[2])
            lower <= upper || continue
            push!(raw, (
                lower=lower,
                upper=upper,
                detected_types=Set((Symbol(special.type),)),
            ))
        end
        sort!(raw; by=item -> (item.lower, item.upper))
        merged = Any[]
        for item in raw
            if !isempty(merged) &&
                    item.lower <= last(merged).upper + float(merge_gap)
                previous = pop!(merged)
                push!(merged, (
                    lower=min(previous.lower, item.lower),
                    upper=max(previous.upper, item.upper),
                    detected_types=union(
                        previous.detected_types, item.detected_types),
                ))
            else
                push!(merged, item)
            end
        end
        for item in merged
            gamma = (item.lower + item.upper) / 2
            angle = _sax_periodic_schur_bracket_angle(slice, gamma)
            angle_to_pi = isfinite(angle) ? abs(pi - abs(angle)) : NaN
            near_r2 = :pd in item.detected_types ||
                (isfinite(angle_to_pi) &&
                 angle_to_pi <= settings.r2_warning_angle)
            push!(brackets, (
                gamma=float(gamma),
                zeta=float(slice.zeta),
                gamma_lower=float(item.lower),
                gamma_upper=float(item.upper),
                gamma_error=float((item.upper - item.lower) / 2),
                floquet_angle=float(angle),
                angle_to_pi=float(angle_to_pi),
                classification=near_r2 ? :near_r2 : :ns_like,
                detected_types=Tuple(sort!(collect(item.detected_types))),
                status=:guess,
                validated=false,
            ))
        end
    end
    sort!(brackets; by=point -> (point.zeta, point.gamma))
    return brackets
end

function _sax_periodic_schur_component_curve(component)
    isempty(component.curve.gamma) && return nothing
    provisional = component.status != :complete
    return provisional ? merge(component.curve, (
        source=merge(component.curve.source, (
            periodic_schur_seed=true,
            provisional=true,
        )),
    )) : merge(component.curve, (
        source=merge(component.curve.source, (periodic_schur_seed=true,)),
    ))
end

"""
    load_sax_periodic_schur_ns_progress(model_p, directory; profile=:final)

Read completed Periodic-Schur products and fall back to compatible per-zeta
slice files and augmented-continuation checkpoints.  It is safe to call this
while the external runner is writing new atomic cache files.
"""
function load_sax_periodic_schur_ns_progress(
        model_p::NamedTuple,
        directory::AbstractString;
        profile::Symbol=:final,
        settings::Union{Nothing,SaxPeriodicSchurNSSettings}=nothing)
    settings = isnothing(settings) ?
        sax_periodic_schur_ns_settings(profile) : settings
    validation_cache = _load_sax_transition_mechanism_cache(
        joinpath(directory, "seed_validation.jld2"),
        :periodic_schur_seed_validation, model_p, settings)
    slices_directory = joinpath(directory, "slices")
    slices_manifest = _load_sax_transition_mechanism_cache(
        joinpath(slices_directory, "pqz_ns_slice_manifest.jld2"),
        :periodic_schur_ns_slices, model_p, settings)
    slices = Any[]
    if slices_manifest.status == :valid
        append!(slices, slices_manifest.payload.slices)
    else
        for zeta in settings.zeta_values
            loaded = _load_sax_transition_mechanism_cache(
                _sax_periodic_schur_slice_path(slices_directory, zeta),
                :periodic_schur_ns_slice, model_p, settings)
            loaded.status == :valid && push!(slices, loaded.payload)
        end
    end
    sort!(slices; by=slice -> slice.zeta)
    slice_failures = slices_manifest.status == :valid ?
        Any[slices_manifest.payload.failures...] : Any[]

    seed_cache = _load_sax_transition_mechanism_cache(
        joinpath(directory, "refined_seed.jld2"),
        :periodic_schur_ns_seed, model_p, settings)
    source_seed_key = seed_cache.status == :valid &&
        seed_cache.payload.status == :valid ? seed_cache.payload.seed.key : nothing
    augmented_directory = joinpath(directory, "augmented")
    augmented_manifest = _load_sax_transition_mechanism_cache(
        joinpath(augmented_directory, "pqz_augmented_ns_manifest.jld2"),
        :periodic_schur_augmented_ns, model_p, settings)
    curves = Any[]
    rows = Any[]
    completed = Set{Tuple{Symbol,Symbol}}()
    components = augmented_manifest.status == :valid ?
        augmented_manifest.payload.components : Any[]
    for component in components
        curve = _sax_periodic_schur_component_curve(component)
        !isnothing(curve) && push!(curves, curve)
        push!(completed, (component.formulation, component.direction))
        push!(rows, (
            formulation=component.formulation,
            direction=component.direction,
            status=component.status,
            points=length(component.curve.gamma),
            checkpoint_count=component.checkpoint_count,
            provisional=component.status != :complete,
            terminal_error=component.terminal_error,
        ))
    end
    augmented_settings = _sax_periodic_schur_augmented_settings(settings)
    for direction in (:positive, :negative)
        found_direction = any(item -> item[2] == direction, completed)
        for formulation in settings.augmented_formulations
            found_direction && break
            paths = _sax_periodic_schur_augmented_paths(
                augmented_directory, formulation, direction)
            loaded = _load_sax_transition_mechanism_cache(
                paths.result, :periodic_schur_augmented_component,
                model_p, settings)
            if loaded.status == :valid
                component = loaded.payload
                curve = _sax_periodic_schur_component_curve(component)
                !isnothing(curve) && push!(curves, curve)
                push!(rows, (
                    formulation=formulation,
                    direction=direction,
                    status=component.status,
                    points=length(component.curve.gamma),
                    checkpoint_count=component.checkpoint_count,
                    provisional=component.status != :complete,
                    terminal_error=component.terminal_error,
                ))
                found_direction = component.checkpoint_count > 0 ||
                    !isempty(component.curve.gamma)
                continue
            end
            checkpoints = _load_sax_mechanism_ns_checkpoints(
                paths.checkpoints, model_p, augmented_settings;
                formulation=formulation,
                direction=direction,
                source_seed_key=source_seed_key,
            )
            checkpoints.status == :valid || continue
            curve = _sax_provisional_ns_curve(
                checkpoints.records, formulation, direction)
            curve = merge(curve, (
                source=merge(curve.source, (periodic_schur_seed=true,)),
            ))
            push!(curves, curve)
            push!(rows, (
                formulation=formulation,
                direction=direction,
                status=:checkpoint,
                points=length(checkpoints.records),
                checkpoint_count=length(checkpoints.records),
                provisional=true,
                terminal_error=nothing,
            ))
            found_direction = true
        end
    end

    root_locus = _sax_periodic_schur_root_locus(slices)
    root_rows = Any[root for slice in slices for root in slice.roots]
    provisional_brackets = sax_periodic_schur_provisional_brackets(
        slices, settings)
    completed_slices = length(slices)
    expected_slices = length(settings.zeta_values)
    slice_status = if completed_slices == expected_slices &&
            isempty(slice_failures)
        :complete
    elseif completed_slices > 0
        :partial
    elseif slices_manifest.status == :valid && !isempty(slice_failures)
        :failed
    else
        slices_manifest.status
    end
    return (
        profile=profile,
        directory=abspath(directory),
        settings=settings,
        validation=validation_cache.status == :valid ?
            validation_cache.payload : nothing,
        validation_status=validation_cache.status,
        slices=(
            status=slice_status,
            completed=completed_slices,
            expected=expected_slices,
            rows=slices,
            failures=slice_failures,
            roots=root_rows,
            accepted_roots=count(root -> root.accepted, root_rows),
            locus=root_locus,
            provisional_brackets=provisional_brackets,
        ),
        seed=seed_cache.status == :valid ? seed_cache.payload : nothing,
        seed_status=seed_cache.status == :valid ?
            seed_cache.payload.status : seed_cache.status,
        augmented=(
            status=augmented_manifest.status == :valid ?
                augmented_manifest.payload.status :
                isempty(curves) ? augmented_manifest.status : :checkpoint,
            curves=curves,
            rows=rows,
            manifest=augmented_manifest.status == :valid ?
                augmented_manifest.payload : nothing,
        ),
    )
end

"""Overlay the independent PQZ root locus and its augmented NS branches."""
function overlay_sax_periodic_schur_ns!(
        figure,
        progress;
        subplot::Integer=1,
        show_labels::Bool=true,
        compact_labels::Bool=false,
        show_provisional_brackets::Bool=true,
        show_provisional_boundary::Bool=true)
    axis = length(figure.subplots) > 1 ? figure[Int(subplot)] : figure
    if show_provisional_brackets
        for (classification, marker, color, long_label, short_label) in (
                (:near_r2, :square, "#0072B2",
                 "PQZ near-R2 bracket*", "PQZ R2*"),
                (:ns_like, :diamond, "#D55E00",
                 "PQZ NS-like bracket*", "PQZ NS*"))
            brackets = filter(
                point -> point.classification == classification,
                progress.slices.provisional_brackets,
            )
            isempty(brackets) && continue
            scatter!(
                axis,
                [point.gamma for point in brackets],
                [point.zeta for point in brackets];
                xerror=[point.gamma_error for point in brackets],
                marker=marker,
                markercolor=:white,
                markerstrokecolor=color,
                markerstrokewidth=1.8,
                markersize=5.5,
                color=color,
                linewidth=1.3,
                label=show_labels ?
                    (compact_labels ? short_label : long_label) : "",
            )
        end
    end
    if show_provisional_boundary
        maximum_zeta_gap = hasproperty(progress.settings, :zeta_step) ?
            2.1 * float(progress.settings.zeta_step) : 0.021
        overlay_sax_dense_pqz_boundary!(
            axis,
            progress.slices.provisional_brackets;
            classification=:near_r2,
            show_label=show_labels,
            compact_label=compact_labels,
            maximum_zeta_gap=maximum_zeta_gap,
            linestyle=:dash,
        )
        overlay_sax_dense_pqz_boundary!(
            axis,
            progress.slices.provisional_brackets;
            classification=:ns_like,
            color="#D55E00",
            show_label=show_labels,
            compact_label=compact_labels,
            maximum_gamma_jump=0.02,
            maximum_angle_jump=0.08,
            maximum_zeta_gap=maximum_zeta_gap,
            linestyle=:dash,
            show_band=false,
        )
    end
    locus = progress.slices.locus
    if !isempty(locus.gamma)
        plot!(
            axis, locus.gamma, locus.zeta;
            color=:orange,
            linewidth=2.5,
            linestyle=:dash,
            marker=:circle,
            markersize=4,
            label=show_labels ?
                (compact_labels ? "PQZ neutral" : "PQZ accepted neutral roots") : "",
        )
    end
    labeled = false
    for curve in progress.augmented.curves
        provisional = hasproperty(curve.source, :provisional) &&
            curve.source.provisional
        plot!(
            axis, curve.gamma, curve.zeta;
            color=:magenta,
            linewidth=provisional ? 2.0 : 3.0,
            alpha=provisional ? 0.7 : 1.0,
            linestyle=:solid,
            marker=provisional ? :circle : :none,
            markersize=provisional ? 2.2 : 0,
            label=show_labels && !labeled ?
                (compact_labels ? "NS-PQZ" :
                 provisional ? "PQZ NS checkpoint" : "PQZ augmented NS") : "",
        )
        labeled = true
    end
    return figure
end

"""Plot mesh convergence and Collocation-versus-PQZ disagreement."""
function plot_sax_periodic_schur_disagreement(progress)
    validation_rows = isnothing(progress.validation) ? Any[] :
        progress.validation.rows
    pmesh = plot(
        xlabel="collocation mesh",
        ylabel="corrected Floquet growth",
        title="Ordinary high-gamma seed",
        legend=:best,
    )
    if isempty(validation_rows)
        annotate!(pmesh, 0.5, 0.5, text("no seed validation cache", 10))
        xlims!(pmesh, 0, 1)
        ylims!(pmesh, 0, 1)
    else
        x = collect(eachindex(validation_rows))
        labels = ["$(row.mesh[1])x$(row.mesh[2])" for row in validation_rows]
        plot!(pmesh, x,
              [row.floquet_coll.corrected_growth for row in validation_rows];
              marker=:circle, linewidth=2, label="FloquetColl")
        plot!(pmesh, x,
              [row.periodic_schur.corrected_growth for row in validation_rows];
              marker=:diamond, linewidth=2, label="FloquetPQZ")
        hline!(pmesh, [0.0]; color=:black, linewidth=1, label="")
        plot!(pmesh; xticks=(x, labels))
    end

    roots = progress.slices.roots
    pagree = plot(
        xlabel="fixed-zeta root index",
        ylabel="absolute method difference",
        title="FloquetColl versus FloquetPQZ",
        yscale=:log10,
        legend=:best,
    )
    if isempty(roots)
        bracket_count = length(progress.slices.provisional_brackets)
        annotate!(pagree, 0.5, 0.5,
                  text("no validated roots; $(bracket_count) provisional brackets", 10))
        xlims!(pagree, 0, 1)
        ylims!(pagree, 1e-16, 1)
    else
        x = collect(eachindex(roots))
        plot!(pagree, x,
              max.([root.validation.growth_difference for root in roots], eps());
              marker=:circle, linewidth=2, label="growth")
        plot!(pagree, x,
              max.([root.validation.angle_difference for root in roots], eps());
              marker=:diamond, linewidth=2, label="angle")
        hline!(pagree, [progress.settings.method_growth_tolerance];
               color=:gray, linestyle=:dash, label="growth tolerance")
    end
    return plot(pmesh, pagree; layout=(1, 2), size=(1100, 430))
end
