using RealTimeAudioDiffEq, Logging, Glob, ProjectRoot
using TOML

include("rt_sax_control.jl")



function stop_model(mgr::RTSaxSerialManager, source::DESource)
    @info "Stopping test"
    stop_update!(mgr)
    stop_model!(source)
end

function run_test(subject_id="83",fingering="C4",configfile="./rt_sax_configuration.toml")
    
    # Load configuration
    configuration = TOML.parsefile(configfile)    
    instrument = configuration["instrument"]
    nmodes = configuration["nmodes"]
    mpath = @projectroot("src",joinpath("impedances",instrument))
    @info "Reading fingering for " * instrument 
    model_parameters = read_fingerings(mpath)
    ts = configuration["ts"] # fixed time scaling
    gain = configuration["gain"] # fixed gain scaling
    mgr = init_serial(configuration["port"])
    model_p = model_parameters[fingering]

    if configuration["logging"]
        # opening the log file and data file
        prefix =  join(["test", string(time_ns())],"_")
        logfile =  prefix * ".log"
        datafile = prefix * ".dat"
        fullpath = @projectroot("src",joinpath(configuration["data_path"],logfile))
        io_log = open(fullpath, "w")
        logger = SimpleLogger(io_log)
        @info "Starting Logger with file " logfile
        mgr.writedata = true
        mgr.logger = logger
        mgr.datafile = @projectroot("src",joinpath(configuration["data_path"],datafile))
    end
    param_map = map(Tuple,configuration["param_map"])
    source = start_model(param_map, model_p, nmodes, ts, gain)
    return (mgr, source, param_map)
end

#(mgr, source, param_map) = run_test()
#start_update!(mgr, source, param_map)
#stop_model(mgr, source)
