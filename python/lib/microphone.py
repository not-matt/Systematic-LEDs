import time
import numpy as np
import pyaudio
import lib.config as config


def start_stream(callback):
    p = pyaudio.PyAudio()
    id=p.get_default_input_device_info()['index']
    
    frames_per_buffer = int(config.settings["configuration"]["MIC_RATE"] / config.settings["configuration"]["FPS"])
    
    numdevices = p.get_device_count()
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0,i)
        # If you have no idea what your devices are called, uncomment the line below to see
        #print(device_info.get("name"))
        if device_info.get('maxInputChannels')>0:
            if device_info.get('name')==config.settings["configuration"]["MIC_NAME"]:
                id=i
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                       rate=config.settings["configuration"]["MIC_RATE"],
                    input=True,
                    input_device_index = id,
                    frames_per_buffer=frames_per_buffer)
        
    overflows = 0
    prev_ovf_time = time.time()
    while True:
        try:
            y = np.fromstring(stream.read(frames_per_buffer), dtype=np.int16)
            y = y.astype(np.float32)
            callback(y)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                if config.settings["configuration"]["USE_GUI"]:
                    gui.label_error.setText('Audio buffer has overflowed {} times'.format(overflows))
                else:
                    print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()
