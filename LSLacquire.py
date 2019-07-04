from testConnect import connect_to_stream
import threading
# from threading import Thread
import time
import numpy as np


class lslReceiver():
    def __init__(self, _save_info_flag, _save_data_flag):
        self.save_data_flag = _save_data_flag
        col_headings_flag = True

        self.inlet = connect_to_stream()

        if _save_info_flag:
            # Save the inlets info as a text file
            info_file = open('info_file.txt', 'w')
            info_file.write(self.inlet.info().as_xml())
            info_file.close()

        if self.save_data_flag:
            self.data_file = open('data.csv', 'w')
            self.events_file = open('events.csv', 'w')

            if col_headings_flag:
                info = self.inlet.info()
                desc = info.desc()

                # Set up column names, 1st ROW
                self.data_file.write('CH_NAME')

                ch = desc.child("channels").child("channel")
                for i in range(info.channel_count()):
                    self.data_file.write(',' + ch.child_value("label"))
                    ch = ch.next_sibling()
                self.data_file.write(',REL_COUNTER\n')

                # 2nd ROW
                self.data_file.write('CH_NUM,')
                chs = str(range(1, info.channel_count( ) +1))[1:-1]
                self.data_file.write(chs + '\n')

                # 3rd ROW
                self.data_file.write('TIMESTAMP\n')

                # Events file headings
                self.events_file.write('TIMESTAMP,CODE,DESC\n')

        print('Starting data acquisition...')

        self.events = []
        self.data = []
        self.num_samples = 0
        self.last_timestamp = 0
        self.lock = threading.Lock()

    def pull_samples(self):
        chunks, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=350)
        # print(np.asarray(chunks).shape)
        if timestamps:
            try:
                self.lock.acquire(True)
                s = len(timestamps)
                self.num_samples += s
                self.last_timestamp = timestamps[-1]
                temp = np.column_stack((np.array(timestamps), np.array(chunks)))
                temp = temp.tolist()
                self.data += temp
                self.lock.release()
            except:
                self.lock.release()
                raise
        #     print('num_samples: ', s)
        # print('total: ', self.num_samples)
        return np.asarray(chunks)

    def save_event(self, code, desc):
        # update latest timestamp
        self.pull_samples()
        event = [self.last_timestamp, code, desc]
        self.events += [event]
        print(event)
        return

    def clean_up(self):
        print('Cleaning...')
        if self.save_data_flag:
            self.save_data()
            self.data_file.close()

            for chunk in self.events:
                self.events_file.write(str(chunk)[1:-1])
                self.events_file.write('\n')
            self.events_file.close()

        self.inlet.close_stream()
        return

    def save_data(self):
        locked = self.lock.acquire(False)
        if not locked:
            print('Skipping data save...')
            return
        for chunk in self.data:
            # print(chunk)
            self.data_file.write(str(chunk)[1:-1])
            self.data_file.write('\n')
        self.data_file.flush()
        self.data = []
        self.lock.release()
        return

    def receive(self):
        chunks = self.pull_samples()
        # print(chunks)
        if len(self.data) > 1000:
            self.save_data()
        return chunks

if __name__ == '__main__':

    # the main logic
    main = lslReceiver(True, True)
    try:
        while True:
            print(main.receive())
            time.sleep(0.1)
    except KeyboardInterrupt:
        main.clean_up()
