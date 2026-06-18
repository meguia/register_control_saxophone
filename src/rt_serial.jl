using LibSerialPort, Logging       
using Base.Threads: @spawn, @atomic, threadid

#global_logger(ConsoleLogger(stderr, Logging.Debug))

######################################
# RTSaxSerialManager
######################################

"""
RTSaxSerialManager(portname::String, baudrate::Int)
It provides thread-safe, asynchronous serial I/O and update of an external state though a channel.
It is used to read Int 16-bits parameters sent from an Arduino in a specific protocol.
Thread-safety assumptions:
- Julia must be started with ≥2 threads
- `start_reader!`, `stop_reader!`, and `close_serial` may be called from any thread
"""
    
mutable struct RTSaxSerialManager
    portname::String # e.g. "/dev/ttyUSB0" or "COM3"
    baudrate::Int # e.g. 115200 
    writedata::Bool # whether to write data to a file, initialized as false
    chan::Channel{Tuple{Int,Int}} # channel for reading data, Tuple{Int,Int} with buf_size 
    @atomic reader_stop::Bool # whether to stop the reader task, initialized as false 
    @atomic updater_stop::Bool # whether to stop the updater task, initialized as false
    reader_task::Task   # task for reading data from the serial port, initialized as nothing 
    updater_task::Task  # task for updating the external state, initialized as nothing
    datafile::String # file to write data to, if writedata is true, initialized as empty string
    datastream::Union{Nothing, IOStream} # stream for writing data to the file, if writedata is true, initialized as nothing
    logger::Union{Nothing,SimpleLogger} # logger for logging messages, initialized as nothing
    RTSaxSerialManager(portname::String, baudrate::Int; buf_size::Int=32) = new(portname,baudrate,false,
        Channel{Tuple{Int,Int}}(buf_size),false,false,Task(nothing),Task(nothing),"",nothing,nothing)
end

# Backward-compatible alias when the name is free in this module.
if !isdefined(@__MODULE__, :SerialManager)
    const SerialManager = RTSaxSerialManager
end

@inline function decode_frame(high::UInt8, mid::UInt8, low::UInt8)
    """decode_frame(high, mid, low)
    Decodes a frame from the shift register.
    The shift register is expected to be of size or 3 bytes high, mid, low = 
    [zzzzS123, yy456789, xxABCDEF] encoding an Int16 value in a specific format:
    the last 6 bits of the low and mid bytes and the last 4 bits of the high byte encode the 16-bit value.
    The MSB S encodes de sign and depending on the this value the bits 1-F are decoded using the two's 
    complement or not.
    The id is obtained from the byte high, high nibble zzzz and starts counting at 1.
    xx and yy are unused bits.
    """
    id = Int((high >> 4) & 0x0F) + 1
    if iszero(high & 0x08)
        # positive Int16
        val =  Int(high & 0x07)*4096 + Int(mid & 0x3F)*64 + Int(low & 0x3F)
    else
        # negative Int16, we need to flip the bits and add 1 to get the two's complement
        val =  -(Int(~high & 0x07)*4096 + Int(~mid & 0x3F)*64 + Int(~low & 0x3F) + 1)
    end
    return id, val
end

function _serial_reader(mgr::RTSaxSerialManager)
    """
    _serial_reader(mgr::RTSaxSerialManager)
    Reads data from the serial port and updates the channel in the RTSaxSerialManager.
    It reads data from the serial port in a loop until stopped.
    It uses a fixed-size byte shift register to accumulate bytes until a full frame is received.
    The frame is expected to end with a byte 0xFF, which indicates the end of the frame and is not part of the data.
    The data is expected to be in a specific format and are decoded using the decode_frame function.
    The id and value are then pushed to the channel.
    """
    local sp = nothing
    if mgr.writedata
        try
            mgr.datastream = open(mgr.datafile, "w") # open the file for writing data
        catch e
            @error "Cannot open log file: $(e)" mgr.datafile
            return
        end    
    end
    #@info "Serial Reader started from thread $(threadid())"
    try
        sp = LibSerialPort.open(mgr.portname, mgr.baudrate)
        #@info "Opened serial: $(mgr.portname) @ $(mgr.baudrate)"
        LibSerialPort.sp_flush(sp, LibSerialPort.SP_BUF_INPUT)
        # fixed byte shift register, no allocations:
        v1, v2, v3 = UInt8(0), UInt8(0), UInt8(0)
        while !(@atomic mgr.reader_stop)
             if bytesavailable(sp) > 0
                # read once into b and test for frame‐end (0xFF)
                if (b = read(sp, UInt8)) == 0xFF
                    # full frame is in (v1,v2,v3)
                    id, val = decode_frame(v1, v2, v3)
                    put!(mgr.chan, (id, val))
                    if mgr.writedata
                        t = time_ns()  # UInt64 timestamp
                        write(mgr.datastream, UInt64(t))
                        write(mgr.datastream, UInt8(id))
                        write(mgr.datastream, Int16(val))
                    end
                else
                    # rotate the register in one tuple‐assignment
                    (v1, v2, v3) = (v2, v3, b)
                end
            else
                sleep(1e-4)
            end
        end
    catch e
        @error "Cannot read serial port: $(e)"
        return
    finally
        if sp !== nothing
             LibSerialPort.close(sp)
             #@info "Serial port closed"
        end
        #@info "Serial reader stopped"
    end    
end

function start_reader!(mgr::RTSaxSerialManager)
    """
    start_reader!(mgr::RTSaxSerialManager)
    Starts the serial reader task.  The task will run in a loop until stopped.
    The task will read data from the serial port and push it to the channel.
    """
    @atomic mgr.reader_stop = false
    mgr.reader_task = @spawn _serial_reader(mgr)
    return nothing
end

function stop_reader!(mgr::RTSaxSerialManager)
    """
    stop_reader!(mgr::RTSaxSerialManager)
    Stops the serial reader task.  The task will stop reading data from the serial port.
    """
    @atomic mgr.reader_stop = true
    return nothing
end

function close_serial(mgr::RTSaxSerialManager)
    """
    close_serial(mgr::RTSaxSerialManager)
    Gracefully closes the serial port and stops the reader task.
    It waits for the reader task to finish before returning.
    """
    stop_reader!(mgr)
    wait(mgr.reader_task)
    @info raw"RTSaxSerialManager closed"
    return nothing
end
