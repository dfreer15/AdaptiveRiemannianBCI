###########################################################
### Simple dummy stream with random 2 channel data at
### 100Hz


import time, ctypes
from random import random as rand
from pylsl import StreamInfo, StreamOutlet

freqHz = 250

def lsl_dummy_stream():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo('DummyStream' + str(freqHz) + 'Hz', 'EEG', 22, freqHz, 'float32', 'myuid342345')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    try:
        print("Press Ctrl+C to terminate")
        while True:
            # make a new random 22-channel sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            mysample = [rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(),
                        rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand()]
            # now send it and wait for a bit
            outlet.push_sample(mysample)
            time.sleep(1.0/freqHz)
    except KeyboardInterrupt:
        del outlet
        print("Loop terminated, shutting stream down")
        time.sleep(0.5)
    return    

if __name__ == '__main__':
    # ctypes.windll.kernel32.SetConsoleTitleA("lslDummyStream")
    lsl_dummy_stream()