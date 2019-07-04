###########################################################
### This program just discovers available streams on the
### network with a TIMEOUT timeout and prints them
### import to use just func, or run from command
### for infinite refreshes


import ctypes
from pylsl import resolve_stream

def get_n_show_streams():
    TIMEOUT = 1.0
    streams = resolve_stream(TIMEOUT)
    if len(streams) == 0:
        print('No streams are available. Make sure a stream is discoverable (or try increasing this functions timeout).')
        return []
    print('Streams found:')
    for idx,stream in enumerate(streams):
        print('{}: {} ({}) - type {}, {} CH, {} Hz'.format(idx, stream.name(), stream.hostname(), stream.type(), stream.channel_count(), stream.nominal_srate()))
    return streams

if __name__ == '__main__':
    ctypes.windll.kernel32.SetConsoleTitleA("showLSLStreams")
    try:
        while True:
            get_n_show_streams()
            input('Press enter to refresh...')
    except KeyboardInterrupt:
        print('Goodbye')