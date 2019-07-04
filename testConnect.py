###########################################################
### Connects to stream, again import to get funcionality
### or run from command to just use as test


import ctypes, time
from pylsl import StreamInlet
from LSLscripts.showStreams import get_n_show_streams
# from showStreams import get_n_show_streams

def connect_to_stream():
    print("Looking for LSL streams...")
    streams = get_n_show_streams()
    while streams == []:
        input("Press Enter to try again or CTRL-C to exit")
        streams = get_n_show_streams()

    # while True:
        # streamChoice = input('Enter stream to connect to ' + ('(0): ' if len(streams) == 1 else ('(0-' + str(len(streams)-1) + '): ')))
    for stream in streams:

        if stream.name()[0] is 'N':
            streamChoice=stream

        # streamChoice = 0

        # if streamChoice.split()[0] == 'ref':
        #     streams = get_n_show_streams()
        #     continue
        # try:
        #     streamChoice = int(streamChoice)
        # except ValueError:
        #     print('Enter integer number literal')
        #     continue
        # if streamChoice in range(len(streams)):
        #     break
        # else:
        #     print('Stream choice out of bounds ' + ('(0): ' if len(streams) == 1 else ('(0-' + str(len(streams)-1) + '): ')))

    # inlet = StreamInlet(streams[streamChoice])
    inlet = StreamInlet(streamChoice)
    print("Connected successfully")
    return inlet

if __name__ == '__main__':
    # ctypes.windll.kernel32.SetConsoleTitleA("testConnect")
    inlet = connect_to_stream()
    inlet.close_stream()
    print("Connection closed")
    time.sleep(1)
    del inlet
