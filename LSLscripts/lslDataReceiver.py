from pylsl import StreamInlet, resolve_stream
import time

print("Looking for an EEG stream...")
streams = resolve_stream('type', 'signal')
print("Found stream")

inlet = StreamInlet(streams[0])

# Save the inlets info as a text file
info_file = open("info_file.txt", "w")
info_file.write(inlet.info().as_xml())
info_file.close()

print("Starting data acquisition...")

numSamples = 0

while True:
    # TODO: Figure out max_samples!!
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=150)
    if timestamps:
        s = len(timestamps)
        numSamples += s
        print("numSamples: ", s)
    print("total: ", numSamples)
    time.sleep(1)