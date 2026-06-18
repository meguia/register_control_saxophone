using RealTimeAudioDiffEq, Logging, Glob, ProjectRoot
using TOML, JSON3, JLD2, REPL, Dates

include("rt_sax_control.jl")

const TERM = REPL.Terminals.TTYTerminal(get(ENV, "TERM", ""), stdin, stdout, stderr)

# ───────────────────────── HANDLE  OF RT SAX ─────────────────────────
mutable struct RTHandle
    mgr::Union{RTSaxSerialManager, Nothing}
    source::Union{DESource, Nothing}
    param_map::Union{Vector{Tuple{Int64,Int64,Float64, Float64}}, Nothing}
    logger::Union{SimpleLogger, Nothing}
    running::Bool
    RTHandle() = new(nothing, nothing, nothing, nothing, false)
end
# ──────────────────────────── USER INTERFACE ────────────────────────────
function readkey_upper()
    """read a single key from stdin, return as uppercase Char
    """
    REPL.Terminals.raw!(TERM, true)
    try
        c = read(stdin, Char)
        return uppercase(c)
    finally
        REPL.Terminals.raw!(TERM, false)
    end
end

function print_help()
    """print_help()
    Prints help message for the experiment controls (in Spanish).
    """
    println()
    println("Controles:")
    println("  B  -> comenzar bloque")
    println("  0-5 -> inicio de la tarea en el orden en que figuran en la partitura")
    println("         (0=practica, 1=primera linea, 2=segunda linea, ...)")
    println("  S  -> fin de la tarea")
    println("  P  -> (opcional) finaliza tarea e imprime resumen de datos")
    println("  E  -> terminar bloque actual")
    println("  H  -> ayuda")
    println("  Q  -> salir")
    println()
end

function show_score(b::Dict{String,Any}, idx::Int, subject_id::String="")
    """ Show the score for the current block.
    """
    if b["task"][2] == "LegatoAsc"
        score_type = "(1)"
    else
        score_type = "(2)"
    end    
    println("────────────────────────────────────────")
    println("Subject: $(subject_id), Block $(idx):")
    println("  type     : ", b["type"])
    println("  fingering: ", b["fingering"], " ", score_type)
    println("  task     : ", join(b["task"], " – "))
    println("────────────────────────────────────────")
end

function list_subjects(json_path::String="./subjects_config.json")
    """list_subjects(json_path::String)
    Lists the available subject IDs from the given JSON file.
    """
    data = JSON3.read(read(json_path, String))
    subjects = haskey(data, "subject") ? data["subject"] : data[:subject]
    return  String.(collect(keys(subjects)))
end

function load_blocks(data, subject_id::String)
    """load_blocks(data, subject_id::String)
    Loads the blocks for the given subject ID from the provided JSON data.
    """
    subjects = data["subject"]
    haskey(subjects, subject_id) || error("The subject $subject_id was not found in the JSON")
    sbj = subjects[subject_id]
    haskey(sbj, "block") || error("The subject $subject_id does not contain the 'block' key")
    blocks_json = sbj["block"]

    blocks = Vector{Dict{String,Any}}()
    for b in blocks_json
        push!(blocks, Dict(
            "type" => String(b["type"]),
            "fingering" => String(b["fingering"]),
            "task" => [String(x) for x in b["task"]]
        ))
    end
    return blocks
end

# ──────────────────────────── RUN A BLOCK ────────────────────────────

function run_block(subject_id::String, block_idx::Int; json_path::String="./subjects_config.json",configfile="./rt_sax_configuration.toml")
    """run_block(subject_id::String, block_idx::Int; json_path::String="./subjects_config.json",configfile="./rt_sax_configuration.toml")
    Runs a single block session for the given subject ID and block index.
    Parameters:
    - subject_id: ID of the subject (as a string).
    - block_idx: Index of the block to run (1-based).
    - json_path: Path to the JSON file containing subject configurations (default: "./subjects_config.json").
    - configfile: Path to the TOML configuration file for the saxophone and serial (default: "./rt_sax_configuration.toml").
    """
    
    # Validations
    if block_idx < 1 || block_idx > 4
        error("block_idx must be between 1 and 4 (inclusive). Got $block_idx.")
    end
    if !(subject_id in list_subjects(json_path))
        ids = list_subjects(json_path)
        @warn "Subject $subject_id does not exist."
        println("Available subjects are: ", join(ids, ", "))
        return
    end    
    
    ## Loads data ##################################################
    # Loads JSON with experiment configuration
    data = JSON3.read(read(json_path, String))
    # Loads configuration file for saxophone and serial
    configuration = nothing
	try
    	io_toml = open(@projectroot("src",joinpath(configfile)), "r")  # Opens the file for reading
    	configuration = TOML.parse(io_toml)
    	close(io_toml)  # Close the file when done
	catch e
    	println("Error opening the file $configfile: ", e)
    end    
    if configuration === nothing
        error("Could not load configuration from $configfile")
    end

    println("Starting a single block session for Subject $subject_id, Block $block_idx")
    # Loads model parameters for the selected instrument (all fingerings)
    instrument = configuration["instrument"]
    nmodes = configuration["nmodes"]
    mpath = @projectroot("src",joinpath("impedances",instrument))
    model_parameters = read_fingerings(mpath)
    ts = configuration["ts"] # fixed time scaling

    # handle for the real-time saxophone (source, serial manager, logger visible everywhere)
    handle = RTHandle()
    handle.param_map = map(Tuple,configuration["param_map"])
     # Loads blocks for the selected subject
    blocks = load_blocks(data, subject_id)
    # print_help() not necessary
    b = blocks[block_idx]
    # reads fingering parameters
    @info "Reading fingering for " * instrument 
    haskey(model_parameters, b["fingering"]) || error("Fingering $(b["fingering"]) not found in model parameters")
    model_p = model_parameters[b["fingering"]]
    # Gain depends on type of block
    if b["type"] == "Real"
        gain = 0.0 # no audio output for real instrument
    else
        gain = configuration["gain"] # gain for the audio output
    end 

    # Starts logging ############

    # opening the log file and writing block info
    prefix =  join(["S"*subject_id, b["type"], b["fingering"], string(time_ns())],"_")
    logfile = prefix * ".log"
    fullpath = @projectroot("src",joinpath(configuration["data_path"],logfile))
    io_log = open(fullpath, "w")
    handle.logger = SimpleLogger(io_log)
    @info "Starting Logger with file " logfile
    with_logger(handle.logger) do
        @info "Subject", subject_id
        @info "Instrument", instrument
        @info "block", block_idx
        @info "type", b["type"]
        @info "fingering", b["fingering"]
    end
    # Main csv log for the whole experiment
    # It contains one entry per block, with subject, type, fingering and date
    logexperiment = @projectroot("src",joinpath(configuration["data_path"],configuration["logexperiment"]))
    open(logexperiment, "a") do io
        # Join the data with commas and add a newline character
        if filesize(logexperiment) == 0
            header = ["subject","type","fingering","block","logfile","date"]
            write(io, join(header, ",") * "\n")
        end
        row_data = [subject_id, b["type"], b["fingering"], block_idx, logfile, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")]
        row_string = join(row_data, ",") * "\n"
        write(io, row_string)
    end

    # Starts Model #################################################
    handle.source = start_model(handle.param_map, model_p, nmodes, ts, gain)
    handle.running = false
    sleep(1.0) # wait a bit to ensure everything is ready

    # complete the log ###########################################
    with_logger(handle.logger) do
        @info "source", handle.source
        @info "nmodes", nmodes
        @info "param_map", handle.param_map
    end
    ###############################################################
    tasks = b["task"]::Vector{String}
    show_score(b, block_idx, subject_id)
    println(tasks)
    println("Start tasks with '0'-'5', 'S' (or 'P') to finish , Q to quit")

    ###########################################################
    # Here starts the main loop for the experiment
    while true
        k = readkey_upper()
        if !handle.running
            # IDLE: accepts 0-5 or Q
            if k == 'Q'
                try
                    @info "Closing log file"
                    close(handle.logger.stream)
                    @info "Stopping model"
                    stop_model!(handle.source)
                    println("done.")
                catch
                    # if it already finished, we ignore
                end
                return
            elseif '0' ≤ k ≤ '5'
                key_digit = parse(Int, k) + 1  # 1..6
                task_label = tasks[key_digit]
                # Creating the serial manager
                handle.mgr = init_serial(configuration["port"])
                handle.mgr.writedata = configuration["logging"]
                # data file for storing sensor values
                prefix =  join(["S"*subject_id, b["type"], b["fingering"], task_label, string(time_ns())],"_")
                handle.mgr.datafile = @projectroot("src",joinpath(configuration["data_path"],prefix*".dat"))
                with_logger(handle.logger) do
                    @info "task", task_label
                    @info "datafile", handle.mgr.datafile
                end
                handle.running = true
                start_update!(handle.mgr, handle.source, handle.param_map)
                println("\a", "Starting task $(key_digit-1) = $task_label")
            else
                # Other keys in idle are silently ignored
            end
            
        else
            # RUNNING: it only accepts S , P or Q
            if k == 'S'
                stop_update!(handle.mgr)
                sleep(0.05) # wait a bit to ensure all is written
                println("───────────────────────────────────────────")
                handle.running = false
                println("Start tasks with '0'-'5', 'S' (or 'P') to finish , Q to quit")
            elseif k == 'P'
                stop_update!(handle.mgr)
                sleep(0.05) # wait a bit to ensure all is written
                println("───────────────────────────────────────────")
                handle.running = false
                if configuration["logging"]
                    println("Summarizing data so far…")
                    summarize_data(handle.mgr, handle.param_map)
                else
                    println("Logging is disabled; no data to summarize.")
                end
                println("───────────────────────────────────────────")
                println("Start tasks with '0'-'5', 'S' (or 'P') to finish , Q to quit")
            elseif k == 'Q'
                stop_update!(handle.mgr)
                sleep(0.05) # wait a bit to ensure all is written
                println("───────────────────────────────────────────")
                handle.running = false
                try
                    @info "Closing log file"
                    close(handle.logger.stream)
                    @info "Stopping model"
                    stop_model!(handle.source)
                    println("done.")
                catch
                    # if it already finished, we ignore
                end
                return
            else
                # Ignore everything else while there is a running task
            end
        end
    end
    ###########################################################
    # Should never reach here
    return nothing
end


