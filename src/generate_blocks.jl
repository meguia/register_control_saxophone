using Random
using JSON3

# ─────────────────────────Task order ─────────────────────────────
const TASK_ORDER_A = ["Practice", "LegatoAsc", "LegatoDesc", "NonlegatoAsc", "NonlegatoDesc", "Overtone"]
const TASK_ORDER_B = ["Practice", "NonlegatoAsc", "NonlegatoDesc", "LegatoAsc", "LegatoDesc", "Overtone"]

"Devuelve :MMRR o :RRMM alternando/contrabalanceando."
function make_type_schemes(nsubjects::Int; typescheme_1 = :MMRR, typescheme_2 = :RRMM)
    half = nsubjects ÷ 2
    schemes = vcat(fill(typescheme_1, half), fill(typescheme_2, nsubjects - half))
    shuffle!(schemes)
    return schemes
end

function make_types_array(typescheme::Symbol)
    s = String(typescheme)
    typescheme_array =[]
    for t in s
        if t == 'M'
            push!(typescheme_array, "Model")
        elseif t == 'R'
            push!(typescheme_array, "Real")
        else
            error("unrecognized symbol $t")
            return
        end
    end

    return typescheme_array
end

"Genera los 4 bloques de un sujeto dado un esquema (:MMRR o :RRMM). Fingering se randomiza en cada par {1,2} y {3,4}."
function make_blocks_for_subject(typescheme::Symbol)
    # tipos por bloque
    types = make_types_array(typescheme)

    # fingering en cada par: permutación aleatoria de ["C4","G4"]
    # pair1 = shuffle(["D4","Dx4"])  # para bloques 1 y 2
    # pair2 = shuffle(["D4","Dx4"])  # para bloques 3 y 4
    # fingerings = [pair1[1], pair1[2], pair2[1], pair2[2]]

    fingerings = ["Dx4", "Dx4", "Dx4", "Dx4"]

    # Inicializamos sin tasks (se asignan globalmente después)
    blocks = Vector{Dict{String,Any}}(undef, 4)
    for i in 1:4
        blocks[i] = Dict(
            "type"      => types[i],
            "fingering" => fingerings[i],
            "task"      => String[]  # placeholder
        )
    end
    return blocks
end

"IDs únicos de sujetos como strings, en [11,99]."
function make_subject_ids(nsubjects::Int)
    @assert nsubjects ≤ 89 "No hay suficientes IDs únicos en [11,99] para $nsubjects sujetos."
    pool = collect(11:99)
    shuffle!(pool)
    return string.(pool[1:nsubjects])
end

"Asigna las dos variantes de tareas globalmente contrabalanceadas a todos los bloques."
function assign_tasks_globally!(all_subjects_blocks::Vector{Vector{Dict{String,Any}}})
    total_blocks = length(all_subjects_blocks) * 4
    @assert total_blocks % 2 == 0
    # Mitad A, mitad B; luego randomizamos
    labels = vcat(fill(:A, total_blocks ÷ 2), fill(:B, total_blocks ÷ 2))
    shuffle!(labels)

    k = 1
    for subj_blocks in all_subjects_blocks
        for i in 1:4
            subj_blocks[i]["task"] = (labels[k] === :A) ? copy(TASK_ORDER_A) : copy(TASK_ORDER_B)
            k += 1
        end
    end
    return nothing
end

"""
Genera la estructura JSON:
{
  "subject": {
    "83": { "block": [ {type=..., fingering=..., task=[...]}, ... ] },
    "84": { "block": [ ... ] },
    ...
  }
}

Parametros:
- nsubjects: cantidad de sujetos (≥16).
- seed: (opcional) semilla para reproducibilidad.
- outfile: nombre de archivo de salida (por defecto "sujetos_config.json").
"""
function generate_counterbalanced_json(; nsubjects::Int=16, seed::Union{Nothing,Int}=nothing, outfile::String="subjects_config.json", pretty::Bool=true, typescheme_1, typescheme_2)
    @assert nsubjects ≥ 16 "Se requieren al menos 16 sujetos."
    if seed !== nothing
        Random.seed!(seed)
    end

    # Esquemas de tipo por sujeto (contrabalanceado)
    schemes = make_type_schemes(nsubjects; typescheme_1, typescheme_2)

    # IDs de sujetos (strings) aleatorios y únicos en [11,99]
    subj_ids = make_subject_ids(nsubjects)

    # Construir bloques por sujeto (sin tasks todavía)
    # Guardamos el vector de bloques por sujeto para poder asignar tasks globalmente
    all_blocks = Vector{Vector{Dict{String,Any}}}(undef, nsubjects)
    for i in 1:nsubjects
        all_blocks[i] = make_blocks_for_subject(schemes[i])
    end

    # Asignación global de tasks (50% A, 50% B en el total de bloques)
    assign_tasks_globally!(all_blocks)

    # Armar estructura final
    subject_obj = Dict{String,Any}()
    for (i, sid) in enumerate(subj_ids)
        subject_obj[sid] = Dict("block" => all_blocks[i])
    end
    root = Dict("subject" => subject_obj)

    # Escribir archivo
    open(outfile, "w") do io
        if pretty && isdefined(JSON3, :pretty)
            JSON3.pretty(io, root)
        else
            JSON3.write(io, root)
        end
    end

    return root, subj_ids
end
