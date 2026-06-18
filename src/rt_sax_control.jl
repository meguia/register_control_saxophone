using RealTimeAudioDiffEq, Logging, Glob, JLD2, TOML, ProjectRoot
using Base.Threads: @spawn, @atomic
using Atomix # for atomic operations on arrays Atomix.@atomic


# In notebook workspaces, names can be imported from previous runs.
# Avoid redeclaring RTSaxSerialManager/serial helpers if they are already available.
if !(isdefined(@__MODULE__, :RTSaxSerialManager) &&
     isdefined(@__MODULE__, :decode_frame) &&
     isdefined(@__MODULE__, :_serial_reader))
    include("./rt_serial.jl")
end

###################################################
# Serial communication and parameter update #
###################################################

function mapvals(x, (a, b, c, d))
    if x < a
        return c
    end
    if x > b
        return d
    end    
    return (x - a) * (d - c) / (b - a) + c
end    

function _param_updater(mgr::RTSaxSerialManager, source::DESource, param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})
    """
    _param_updater(source::DESource, mgr::RTSaxSerialManager, param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})
    Updates the parameters of the source based on the data received from the channel.
    The id and value are received from the channel, and the value is mapped to the range
    specified in param_map. The id is used to update the corresponding parameter in the source
    (a specific one-to-one mapping between id and parameter is assumed).
    """
    #@info "Parameter updater started from thread $(threadid())"
    sleep(0.1) 
    try
        while true
            if @atomic mgr.updater_stop
                break
            end
            # read from the channel
            if isready(mgr.chan)
                id, val = take!(mgr.chan)
                if id <= 2
                    set_param!(source, id, mapvals(val, param_map[id]))
                else
                    @warn "Unexpected channel ID" id=id
                end
            else
                sleep(1e-4)
            end
        end
    catch e
        if e isa InvalidStateException
            @info "channel closed => normal shutdown"
        else
            @error "Parameter updater error" error=e
        end
    finally
        @info "Parameter updater stopped"
    end    
end


# Update the parameters of the source based on the data received from the channel.

function start_update!(mgr::RTSaxSerialManager, source::DESource, param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})   
    """
    start_update!(mgr::RTSaxSerialManager, source::DESource, param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})
    Starts the serial reader and parameter updater tasks.
    The serial reader task reads data from the serial port and pushes it to the channel.
    The parameter updater task reads from the channel and updates the parameters of the source.
    """
    
    #@info "Starting serial reader task"
    @atomic mgr.reader_stop = false
    mgr.reader_task = @spawn _serial_reader(mgr)
    #@info "Serial reader task started"
    #@info "Starting parameter updater task"
    @atomic mgr.updater_stop = false
    mgr.updater_task = @spawn _param_updater(mgr, source, param_map)

    return nothing
end

#wait for a task to finish or timeout

function wait_task(t::Task; timeout::Float64=2.0)
    """
    Waits for a task to finish or a timeout to occur.
    """
    done = Ref(false)
    waiter = @async (wait(t); done[] = true)
    t0 = time()
    while !done[] && (time() - t0) < timeout
        sleep(0.01)
    end
    return done[]
end

#stop the serial manager

function stop_update!(mgr::RTSaxSerialManager)
    """
    stop!(mgr::RTSaxSerialManager)
    Stops the serial reader and parameter updater tasks.
    1. The parameter updater task is stopped by setting the updater_stop flag to true.
    2. The serial reader task is stopped by setting the reader_stop flag to true.
    """

    #@info "Stopping parameter updater task"
    @atomic mgr.updater_stop = true
    ok_up = try
        wait_task(mgr.updater_task; timeout=2.0)
    catch
        false
    end
    @info "Updater task stopped: $(ok_up)"
    
    #@info "Stopping serial reader task"
    @atomic mgr.reader_stop = true
    ok_rd = try
        wait_task(mgr.reader_task; timeout=2.0)
    catch
        false
    end
    @info "Reader task stopped: $(ok_rd)"
    #close(mgr.chan) dont close the channel, it is reused
    # Drain the channel
    try
        while isready(mgr.chan)
            take!(mgr.chan)
        end
    catch
    end

    if mgr.writedata
        #@info "Closing data file"
        try; close(mgr.datastream); catch; end
        try; close(mgr.logger.stream); catch; end
    end
    return nothing
end

function init_serial(portname::String)
    """
    Initialize serial port
    portname = "/dev/ttyUSB0" # Linux
    portname = "COM5" # Windows
    portname = "/dev/cu.usbmodem14101" # Mac
    """
    baudrate = 115200 # default baudrate for the saxophone model
    mgr = RTSaxSerialManager(portname, baudrate)
    return mgr
end


function start_param_updater_only!(mgr::RTSaxSerialManager, source::DESource,
                                   param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})
    """
    start_param_updater_only!(mgr::RTSaxSerialManager, source::DESource,
                              param_map::Vector{Tuple{Int64,Int64,Float64, Float64}})
    Starts only the parameter updater task.
    The parameter updater task reads from the channel and updates the parameters of the source.
    """
    @atomic mgr.updater_stop = false
    @info "Starting parameter updater task"
    mgr.updater_task = @spawn _param_updater(mgr, source, param_map)
    @info mgr.updater_task
    return nothing
end

####################
# Utilities
###################

@inline summary(x) = (minimum(x),sum(x)/length(x),maximum(x),length(x))

function read_data(mgr)
    """
    Reads the data from the mgr.datafile and returns the timestamps and values for each channel.
    """
    f = open(mgr.datafile, "r")
	t1 = Int64[]
	t2 = Int64[]
	val1 = Int16[]
	val2 = Int16[]
	while !eof(f)
		t = read(f,Int64)
		id = read(f,UInt8)
		val = read(f,Int16)
		if id == 1
			push!(t1,t)
			push!(val1,val)
		elseif id == 2
			push!(t2,t)
			push!(val2,val)
		end
	end	
	close(f)
	return t1, t2, val1, val2
end		

function read_data(datafile::String)
    """
    Reads the data from the datafile and returns the timestamps and values for each channel.
    """
    f = open(datafile, "r")
	t1 = Int64[]
	t2 = Int64[]
	val1 = Int16[]
	val2 = Int16[]
	while !eof(f)
		t = read(f,Int64)
		id = read(f,UInt8)
		val = read(f,Int16)
		if id == 1
			push!(t1,t)
			push!(val1,val)
		elseif id == 2
			push!(t2,t)
			push!(val2,val)
		end
	end	
	close(f)
	return t1, t2, val1, val2
end		

function summarize_data(mgr,param_map)
    """
    Summarizes the data read from the log file.
    It reads the data from the log file, computes the duration of the recording,
    and computes the summary statistics for the input values and mapped parameters.
    """
    t1, t2, val1, val2 = read_data(mgr)
    duration = (t1[end] - t1[1]) / 1e9 # convert to seconds
    sval1 = summary(val1)
    sval2 = summary(val2)
    γ = mapvals.(val1, Ref(param_map[1]))
    ζ = mapvals.(val2, Ref(param_map[2]))
    sγ = summary(γ)
    sζ = summary(ζ)
    @info "Data summary: duration=$(duration) seconds, number of samples=$(length(t1))"
    @info "Input Value 1:   min=$(sval1[1]), max=$(sval1[3]), mean=$(sval1[2]), count=$(sval1[4])"
    @info "Input Value 2:   min=$(sval2[1]), max=$(sval2[3]), mean=$(sval2[2]), count=$(sval2[4])"
    @info "Mapped Parameter γ: min=$(sγ[1]), max=$(sγ[3]), mean=$(sγ[2]), count=$(sγ[4])"
    @info "Mapped Parameter ζ: min=$(sζ[1]), max=$(sζ[3]), mean=$(sζ[2]), count=$(sζ[4])"
    return nothing

end


################################################
# ODE source control functions
################################################

function stop_model!(source::DESource)
    """
    Stops the ODE simulation task.
    """
    
    @info "Stopping ODE simulation task"
    try
        stop_DESource(source)
        @info "ODE simulation task stopped"
    catch e
        @warn "stop_DESource: Wait timed out" error=e
    end
    return nothing
end


function start_model(param_map::Vector{Tuple{Int64,Int64,Float64, Float64}}, model_p::NamedTuple, nmodes::Int64,ts::Float64, gain::Float64)
    """
    Starts the ODE simulation task with the given parameters.   
    param_map: Vector of tuples with (input_min, input_max, param_min, param_max)
    model_p: NamedTuple with model parameters (α, ω, C) as vectors
    nmodes: Number of modes to simulate (8)
    ts: time scaling factor
    gain: Output gain
    """
     # it assumes that the parameters start from the lower value of the range
    pars = set_parameters(param_map[1][3],param_map[2][3],model_p,nmodes)
    u0 = vcat([-pars[1], 0],repeat([0.0,0.001],nmodes))

    # create the ODE source
    parray = collect(3:2:2*nmodes+2) # sum of the modes (L)
    source = DESource(saxRN!, u0,pars; channel_map = [parray])
    
    # get default output device
    output_device = get_default_output_device();    
    @info "Starting the ODE simulation task"
    start_DESource(source, output_device; buffer_size=convert(UInt32,1024))
    @atomic source.data.control.ts = ts
    @atomic source.data.control.gain = gain
    return source
end





##################
# Saxophone model
##################

function read_fingerings(mpath::String)
    """
    Read fingering data from JLD2 files in the specified directory.
    """
    mpfiles = glob("*.jld2",mpath)
    model_parameters = Dict{String,NamedTuple}()
    for file in mpfiles
        data = load_object(file)
        key = splitext(splitpath(file)[end])[1]
        model_parameters[key] = data
    end
    return model_parameters
end

function set_parameters(γ::Float64, ζ::Float64, model_p::NamedTuple, nmodes::Int64)
    """
    Set the parameters for the saxophone model.
    γ: Blowing pressure
    ζ: Lip reed opening
    model_p: NamedTuple with model parameters (α, ω, C) as vectors
    nmodes: Number of modes to simulate (8)
    """
    nparfix = 2
    pars = zeros(nparfix+3*nmodes)
    pars[1:nparfix] = [γ,ζ]
    pars[nparfix+1:nparfix+nmodes] = model_p.α[1:nmodes]
    pars[nparfix+nmodes+1:nparfix+nmodes*2] = model_p.ω[1:nmodes]
    pars[nparfix+nmodes*2+1:nparfix+nmodes*3] = model_p.C[1:nmodes]
    return pars
end  

# the actual equations of the saxophone model 

function saxRN!(dx,x,p,t)
    nparfix = 2
    nmodes = Int(length(x)/2-1)
    (γ,ζ) = p[1:nparfix]
    α = p[nparfix+1:nparfix+nmodes]
    ω = p[nparfix+nmodes+1:nparfix+nmodes*2]
    C = p[nparfix+nmodes*2+1:nparfix+nmodes*3]
    P = sum(x[3:2:end])
    Fc = 100.0*min(real(x[1])+1,0)^2*(1-x[2])
    u = ζ*max(real(x[1])+1,0)*sign(γ-P)*sqrt(abs(γ-P))
    dx[1] = x[2]
    dx[2] = -4.224*x[2]+17.842176*(P-γ-x[1]+Fc)
    @inbounds for m=1:nmodes
        n = 2*m+1
        dx[n] = -α[m]*x[n]-2*ω[m]*x[n+1]+2*C[m]*u
        dx[n+1] = -α[m]*x[n+1]+0.5*ω[m]*x[n]
    end
    return dx
end   




