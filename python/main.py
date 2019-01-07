from __future__ import print_function
from __future__ import division
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
import time
import sys
import pyaudio
import numpy as np
import lib.config  as config
#import lib.microphone as microphone
import lib.dsp as dsp
#import lib.led as led
import lib.melbank as melbank
import lib.devices as devices
import random
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
if config.settings["configuration"]["USE_GUI"]:
    from lib.qrangeslider import QRangeSlider
    from lib.qfloatslider import QFloatSlider
    import pyqtgraph as pg
    from PyQt5.QtGui import QColor, QIcon

class BoardManager():
    """
    Class that manages all the boards, with their respective Visualisations, GUI tabs, and DSPs
    """
    def __init__(self):
        self.visualizers = {}
        self.boards = {}
        self.signal_processers = {}

    def addBoard(self, board, config_exists=False, req_config=None, gen_config=None):
        if not config_exists:
            self.addConfig(board, req_config, gen_config)
        # Initialise Visualiser
        self.visualizers[board] = Visualizer(board)
        # Initialise DSP
        self.signal_processers[board] = DSP(board)
        # Initialise Device
        if config.settings["devices"][board]["configuration"]["TYPE"] == 'ESP8266':
            self.boards[board] = devices.ESP8266(
                    auto_detect=config.settings["devices"][board]["configuration"]["AUTO_DETECT"],
                       mac_addr=config.settings["devices"][board]["configuration"]["MAC_ADDR"],
                             ip=config.settings["devices"][board]["configuration"]["UDP_IP"],
                           port=config.settings["devices"][board]["configuration"]["UDP_PORT"])
        if config.settings["devices"][board]["configuration"]["TYPE"] == 'PxMatrix':
            self.boards[board] = devices.PxMatrix(
                    auto_detect=config.settings["devices"][board]["configuration"]["AUTO_DETECT"],
                       mac_addr=config.settings["devices"][board]["configuration"]["MAC_ADDR"],
                             ip=config.settings["devices"][board]["configuration"]["UDP_IP"],
                           port=config.settings["devices"][board]["configuration"]["UDP_PORT"])
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'RaspberryPi':
            self.boards[board] = devices.RaspberryPi(
                       n_pixels=config.settings["devices"][board]["configuration"]["N_PIXELS"],
                            pin=config.settings["devices"][board]["configuration"]["LED_PIN"],
                   invert_logic=config.settings["devices"][board]["configuration"]["LED_INVERT"],
                           freq=config.settings["devices"][board]["configuration"]["LED_FREQ_HZ"],
                            dma=config.settings["devices"][board]["configuration"]["LED_DMA"])
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'Fadecandy':
            self.boards[board] = devices.FadeCandy(
                         server=config.settings["devices"][board]["configuration"]["SERVER"])
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'BlinkStick':
            self.boards[board] = devices.BlinkStick()
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'DotStar':
            self.boards[board] = devices.DotStar()
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'sACNClient':
            self.boards[board] = devices.sACNClient(
                            ip = config.settings["devices"][board]["configuration"]["IP"],
                start_universe = config.settings["devices"][board]["configuration"]["START_UNIVERSE"],
                 start_channel = config.settings["devices"][board]["configuration"]["START_CHANNEL"],
                 universe_size = config.settings["devices"][board]["configuration"]["UNIVERSE_SIZE"],
                 channel_count = config.settings["devices"][board]["configuration"]["N_PIXELS"] * 3,
                           fps = config.settings["configuration"]["FPS"])
        elif config.settings["devices"][board]["configuration"]["TYPE"] == 'Stripless':
            self.boards[board] = devices.Stripless()
        if config.settings["configuration"]["USE_GUI"]:
            gui.addBoard(board)

    def delBoard(self, board):
        #print("deleting board {}".format(board))
        del self.visualizers[board]
        del self.signal_processers[board]
        del self.boards[board]
        del config.settings["devices"][board]
        gui.delBoard(board)

    def addConfig(self, board, req_config, gen_config):
        if board in self.boards:
            raise ValueError('Device already exists under name: {}\nPlease use a different name.'.format(board))
        config.settings["devices"][board] = {}
        # Update missing values from defaults
        merged_general_config = {**config.default_general_config, **gen_config}
        # combine into one configuration dict
        merged_config = {**req_config, **merged_general_config}
        # Generate device config dict
        config.settings["devices"][board]["configuration"] = {}
        for configuration in merged_config:
            config.settings["devices"][board]["configuration"][configuration] = merged_config[configuration]
        # Generate device effect opts dict
        config.settings["devices"][board]["effect_opts"] = config.default_effect_opts

class BeatDetector():
    def __init__(self):
        pass

    def update(audio_data):
        pass

class Visualizer(BoardManager):
    def __init__(self, board):
        # Name of board this for which this visualizer instance is visualising
        self.board = board
        # Dictionary linking names of effects to their respective functions
        self.effects = {"Scroll":self.visualize_scroll,
                        "Energy":self.visualize_energy,
                        "Spectrum":self.visualize_spectrum,
                        "Power":self.visualize_power,
                        "Wavelength":self.visualize_wavelength,
                        "Beat":self.visualize_beat,
                        "Wave":self.visualize_wave,
                        "Bars":self.visualize_bars,
                        #"Pulse":self.visualize_pulse,
                        #"Pulse":self.visualize_pulse,
                        #"Auto":self.visualize_auto,
                        "Single":self.visualize_single,
                        "Fade":self.visualize_fade,
                        "Gradient":self.visualize_gradient,
                        "Calibration": self.visualize_calibration}
        # List of all the visualisation effects that aren't audio reactive.
        # These will still display when no music is playing.
        self.non_reactive_effects = ["Single", "Gradient", "Fade", "Calibration"]
        # Setup for frequency detection algorithm
        self.freq_channel_history = 40
        self.beat_count = 0
        self.freq_channels = [deque(maxlen=self.freq_channel_history) for i in range(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"])]
        self.prev_output = np.array([[0 for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])] for i in range(3)])
        self.output = np.array([[0 for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])] for i in range(3)])
        self.prev_spectrum = np.array([config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2])
        self.current_freq_detects = {"beat":False,
                                     "low":False,
                                     "mid":False,
                                     "high":False}
        self.prev_freq_detects = {"beat":0,
                                  "low":0,
                                  "mid":0,
                                  "high":0}
        self.detection_ranges = {"beat":(0,int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.11)),
                                 "low":(int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.13),
                                        int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.4)),
                                 "mid":(int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.4),
                                        int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.7)),
                                 "high":(int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.8),
                                         int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]))}
        self.min_detect_amplitude = {"beat":0.7,
                                     "low":0.5,
                                     "mid":0.3,
                                     "high":0.3}
        self.min_percent_diff = {"beat":70,
                                 "low":100,
                                 "mid":50,
                                 "high":30}
        # Setup for fps counter
        self.frame_counter = 0
        self.start_time = time.time()
        # Setup for "Wave" (don't change these)
        self.wave_wipe_count = 0
        # Setup for "Power" (don't change these)
        self.power_indexes = []
        self.power_brightness = 0

    def get_vis(self, y, audio_input):
        self.update_freq_channels(y)
        self.detect_freqs()
        if config.settings["devices"][self.board]["configuration"]["current_effect"] in self.non_reactive_effects:
            self.prev_output = self.effects[config.settings["devices"][self.board]["configuration"]["current_effect"]]()
        elif audio_input:
            self.prev_output = self.effects[config.settings["devices"][self.board]["configuration"]["current_effect"]](y)
        else:
            self.prev_output = np.multiply(self.prev_output, 0.95)
        self.frame_counter += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.start_time = time.time()
            fps = self.frame_counter//elapsed
            latency = elapsed/self.frame_counter
            self.frame_counter = 0

            if config.settings["configuration"]["USE_GUI"]:
                gui.label_latency.setText("{:0.3f} ms Processing Latency   ".format(latency))
                gui.label_fps.setText('{:.0f} / {:.0f} FPS   '.format(fps, config.settings["configuration"]["FPS"]))
        return self.prev_output

    def _split_equal(self, value, parts):
        value = float(value)
        return [int(round(i*value/parts)) for i in range(1,parts+1)]

    def update_freq_channels(self, y):
        for i in range(len(y)):
            self.freq_channels[i].appendleft(y[i])

    def detect_freqs(self):
        """
        Function that updates current_freq_detects. Any visualisation algorithm can check if
        there is currently a beat, low, mid, or high by querying the self.current_freq_detects dict.
        """
        channel_avgs = []
        differences = []
        for i in range(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]):
            channel_avgs.append(sum(self.freq_channels[i])/len(self.freq_channels[i]))
            differences.append(((self.freq_channels[i][0]-channel_avgs[i])*100)//channel_avgs[i])
        for i in ["beat", "low", "mid", "high"]:
            if any(differences[j] >= self.min_percent_diff[i]\
                   and self.freq_channels[j][0] >= self.min_detect_amplitude[i]\
                            for j in range(*self.detection_ranges[i]))\
                        and (time.time() - self.prev_freq_detects[i] > 0.1)\
                        and len(self.freq_channels[0]) == self.freq_channel_history:
                self.prev_freq_detects[i] = time.time()
                self.current_freq_detects[i] = True
            else:
                self.current_freq_detects[i] = False

    def visualize_scroll(self, y):
        # Effect that scrolls colours corresponding to frequencies across the strip 
        #y = y**1.5
        n_pixels = config.settings["devices"][self.board]["configuration"]["N_PIXELS"]
        y = np.copy(interpolate(y, n_pixels // 2))
        board_manager.signal_processers[self.board].common_mode.update(y)
        diff = y - self.prev_spectrum
        self.prev_spectrum = np.copy(y)

        y = np.clip(y, 0, 1)
        lows = y[:len(y) // 6]
        mids = y[len(y) // 6: 2 * len(y) // 5]
        high = y[2 * len(y) // 5:]
        # max values
        lows_max = np.max(lows)#*config.settings["devices"][self.board]["effect_opts"]["Scroll"]["lows_multiplier"])
        mids_max = float(np.max(mids))#*config.settings["devices"][self.board]["effect_opts"]["Scroll"]["mids_multiplier"])
        high_max = float(np.max(high))#*config.settings["devices"][self.board]["effect_opts"]["Scroll"]["high_multiplier"])
        # indexes of max values
        # map to colour gradient
        lows_val = (np.array(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Scroll"]["lows_color"])) * lows_max).astype(int)
        mids_val = (np.array(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Scroll"]["mids_color"])) * mids_max).astype(int)
        high_val = (np.array(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Scroll"]["high_color"])) * high_max).astype(int)
        # Scrolling effect window
        speed = config.settings["devices"][self.board]["effect_opts"]["Scroll"]["speed"]
        self.output[:, speed:] = self.output[:, :-speed]
        self.output = (self.output * config.settings["devices"][self.board]["effect_opts"]["Scroll"]["decay"]).astype(int)
        self.output = gaussian_filter1d(self.output, sigma=config.settings["devices"][self.board]["effect_opts"]["Scroll"]["blur"])
        # Create new color originating at the center
        self.output[0, :speed] = lows_val[0] + mids_val[0] + high_val[0]
        self.output[1, :speed] = lows_val[1] + mids_val[1] + high_val[1]
        self.output[2, :speed] = lows_val[2] + mids_val[2] + high_val[2]
        # Update the LED strip
        #return np.concatenate((self.prev_spectrum[:, ::-speed], self.prev_spectrum), axis=1)
        if config.settings["devices"][self.board]["effect_opts"]["Scroll"]["mirror"]:
            p = np.concatenate((self.output[:, ::-2], self.output[:, ::2]), axis=1)
        else:
            p = self.output
        return p

    def visualize_energy(self, y):
        """Effect that expands from the center with increasing sound energy"""
        y = np.copy(y)
        board_manager.signal_processers[self.board].gain.update(y)
        y /= board_manager.signal_processers[self.board].gain.value
        scale = config.settings["devices"][self.board]["effect_opts"]["Energy"]["scale"]
        # Scale by the width of the LED strip
        y *= float((config.settings["devices"][self.board]["configuration"]["N_PIXELS"] * scale) - 1)
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        # Map color channels according to energy in the different freq bands
        #y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        diff = y - self.prev_spectrum
        self.prev_spectrum = np.copy(y)
        spectrum = np.copy(self.prev_spectrum)
        spectrum = np.array([j for i in zip(spectrum,spectrum) for j in i])
        # Color channel mappings
        r = int(np.mean(spectrum[:len(spectrum) // 3]**scale)*config.settings["devices"][self.board]["effect_opts"]["Energy"]["r_multiplier"])
        g = int(np.mean(spectrum[len(spectrum) // 3: 2 * len(spectrum) // 3]**scale)*config.settings["devices"][self.board]["effect_opts"]["Energy"]["g_multiplier"])
        b = int(np.mean(spectrum[2 * len(spectrum) // 3:]**scale)*config.settings["devices"][self.board]["effect_opts"]["Energy"]["b_multiplier"])
        # Assign color to different frequency regions
        self.output[0, :r] = 255
        self.output[0, r:] = 0
        self.output[1, :g] = 255
        self.output[1, g:] = 0
        self.output[2, :b] = 255
        self.output[2, b:] = 0
        # Apply blur to smooth the edges
        self.output[0, :] = gaussian_filter1d(self.output[0, :], sigma=config.settings["devices"][self.board]["effect_opts"]["Energy"]["blur"])
        self.output[1, :] = gaussian_filter1d(self.output[1, :], sigma=config.settings["devices"][self.board]["effect_opts"]["Energy"]["blur"])
        self.output[2, :] = gaussian_filter1d(self.output[2, :], sigma=config.settings["devices"][self.board]["effect_opts"]["Energy"]["blur"])
        if config.settings["devices"][self.board]["effect_opts"]["Energy"]["mirror"]:
            p = np.concatenate((self.output[:, ::-2], self.output[:, ::2]), axis=1)
        else:
            p = self.output
        return p

    def visualize_wavelength(self, y):
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        board_manager.signal_processers[self.board].common_mode.update(y)
        diff = y - self.prev_spectrum
        self.prev_spectrum = np.copy(y)
        # Color channel mappings
        r = board_manager.signal_processers[self.board].r_filt.update(y - board_manager.signal_processers[self.board].common_mode.value)
        g = np.abs(diff)
        b = board_manager.signal_processers[self.board].b_filt.update(np.copy(y))
        r = np.array([j for i in zip(r,r) for j in i])
        output = np.array([colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["color_mode"]][0][
                                    (config.settings["devices"][self.board]["configuration"]["N_PIXELS"] if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else 0):
                                    (None if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else config.settings["devices"][self.board]["configuration"]["N_PIXELS"]):]*r,
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["color_mode"]][1][
                                    (config.settings["devices"][self.board]["configuration"]["N_PIXELS"] if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else 0):
                                    (None if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else config.settings["devices"][self.board]["configuration"]["N_PIXELS"]):]*r,
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["color_mode"]][2][
                                    (config.settings["devices"][self.board]["configuration"]["N_PIXELS"] if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else 0):
                                    (None if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_grad"] else config.settings["devices"][self.board]["configuration"]["N_PIXELS"]):]*r])
        #self.prev_spectrum = y
        colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["color_mode"]] = np.roll(
                    colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["color_mode"]],
                    config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["roll_speed"]*(-1 if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["reverse_roll"] else 1),
                    axis=1)
        output[0] = gaussian_filter1d(output[0], sigma=config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["blur"])
        output[1] = gaussian_filter1d(output[1], sigma=config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["blur"])
        output[2] = gaussian_filter1d(output[2], sigma=config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["blur"])
        if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["flip_lr"]:
            output = np.fliplr(output)
        if config.settings["devices"][self.board]["effect_opts"]["Wavelength"]["mirror"]:
            output = np.concatenate((output[:, ::-2], output[:, ::2]), axis=1)
        return output
    
    def visualize_spectrum(self, y):
        """Effect that maps the Mel filterbank frequencies onto the LED strip"""
        #print(len(y))
        #print(y)
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        board_manager.signal_processers[self.board].common_mode.update(y)
        diff = y - self.prev_spectrum
        self.prev_spectrum = np.copy(y)
        # Color channel mappings
        r = board_manager.signal_processers[self.board].r_filt.update(y - board_manager.signal_processers[self.board].common_mode.value)
        g = np.abs(diff)
        b = board_manager.signal_processers[self.board].b_filt.update(np.copy(y))
        r *= config.settings["devices"][self.board]["effect_opts"]["Spectrum"]["r_multiplier"]
        g *= config.settings["devices"][self.board]["effect_opts"]["Spectrum"]["g_multiplier"]
        b *= config.settings["devices"][self.board]["effect_opts"]["Spectrum"]["b_multiplier"]
        # Mirror the color channels for symmetric output
        r = np.concatenate((r[::-1], r))
        g = np.concatenate((g[::-1], g))
        b = np.concatenate((b[::-1], b))
        output = np.array([r, g,b]) * 255
        self.prev_spectrum = y
        return output

    def visualize_auto(self,y):
        """Automatically (intelligently?) cycle through effects"""
        return self.visualize_beat(y) # real intelligent

    def visualize_wave(self, y):
        """Effect that flashes to the beat with scrolling coloured bits"""
        if self.current_freq_detects["beat"]:
            output = np.zeros((3,config.settings["devices"][self.board]["configuration"]["N_PIXELS"]))
            output[0][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_flash"])[0]
            output[1][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_flash"])[1]
            output[2][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_flash"])[2]
            self.wave_wipe_count = config.settings["devices"][self.board]["effect_opts"]["Wave"]["wipe_len"]
        else:
            output = np.copy(self.prev_output)
            #for i in range(len(self.prev_output)):
            #    output[i] = np.hsplit(self.prev_output[i],2)[0]
            output = np.multiply(self.prev_output,config.settings["devices"][self.board]["effect_opts"]["Wave"]["decay"])
            for i in range(self.wave_wipe_count):
                output[0][i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[0]
                output[0][-i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[0]
                output[1][i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[1]
                output[1][-i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[1]
                output[2][i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[2]
                output[2][-i]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Wave"]["color_wave"])[2]
            #output = np.concatenate([output,np.fliplr(output)], axis=1)
            if self.wave_wipe_count > config.settings["devices"][self.board]["configuration"]["N_PIXELS"]//2:
                self.wave_wipe_count = config.settings["devices"][self.board]["configuration"]["N_PIXELS"]//2
            self.wave_wipe_count += config.settings["devices"][self.board]["effect_opts"]["Wave"]["wipe_speed"]
        return output

    def visualize_beat(self, y):
        """Effect that flashes to the beat"""
        if self.current_freq_detects["beat"]:
            output = np.zeros((3,config.settings["devices"][self.board]["configuration"]["N_PIXELS"]))
            output[0][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Beat"]["color"])[0]
            output[1][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Beat"]["color"])[1]
            output[2][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Beat"]["color"])[2]
        else:
            output = np.copy(self.prev_output)
            output = np.multiply(self.prev_output,config.settings["devices"][self.board]["effect_opts"]["Beat"]["decay"])
        return output

    def visualize_bars(self, y):
        # Bit of fiddling with the y values
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        board_manager.signal_processers[self.board].common_mode.update(y)
        self.prev_spectrum = np.copy(y)
        # Color channel mappings
        r = board_manager.signal_processers[self.board].r_filt.update(y - board_manager.signal_processers[self.board].common_mode.value)
        r = np.array([j for i in zip(r,r) for j in i])
        # Split y into [resulution] chunks and calculate the average of each
        max_values = np.array([max(i) for i in np.array_split(r, config.settings["devices"][self.board]["effect_opts"]["Bars"]["resolution"])])
        max_values = np.clip(max_values, 0, 1)
        color_sets = []
        for i in range(config.settings["devices"][self.board]["effect_opts"]["Bars"]["resolution"]):
            # [r,g,b] values from a multicolour gradient array at [resulution] equally spaced intervals
            color_sets.append([colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Bars"]["color_mode"]]\
                              [j][i*(config.settings["devices"][self.board]["configuration"]["N_PIXELS"]//config.settings["devices"][self.board]["effect_opts"]["Bars"]["resolution"])] for j in range(3)])
        output = np.zeros((3,config.settings["devices"][self.board]["configuration"]["N_PIXELS"]))
        chunks = np.array_split(output[0], config.settings["devices"][self.board]["effect_opts"]["Bars"]["resolution"])
        n = 0
        # Assign blocks with heights corresponding to max_values and colours from color_sets
        for i in range(len(chunks)):
            m = len(chunks[i])
            for j in range(3):
                output[j][n:n+m] = color_sets[i][j]*max_values[i]
            n += m
        colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Bars"]["color_mode"]] = np.roll(
                    colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Bars"]["color_mode"]],
                    config.settings["devices"][self.board]["effect_opts"]["Bars"]["roll_speed"]*(-1 if config.settings["devices"][self.board]["effect_opts"]["Bars"]["reverse_roll"] else 1),
                    axis=1)
        if config.settings["devices"][self.board]["effect_opts"]["Bars"]["flip_lr"]:
            output = np.fliplr(output)
        if config.settings["devices"][self.board]["effect_opts"]["Bars"]["mirror"]:
            output = np.concatenate((output[:, ::-2], output[:, ::2]), axis=1)
        return output

    def visualize_power(self, y):
        #config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]
        # Bit of fiddling with the y values
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        board_manager.signal_processers[self.board].common_mode.update(y)
        self.prev_spectrum = np.copy(y)
        # Color channel mappings
        r = board_manager.signal_processers[self.board].r_filt.update(y - board_manager.signal_processers[self.board].common_mode.value)
        r = np.array([j for i in zip(r,r) for j in i])
        output = np.array([colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][0, :config.settings["devices"][self.board]["configuration"]["N_PIXELS"]]*r,
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][1, :config.settings["devices"][self.board]["configuration"]["N_PIXELS"]]*r,
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][2, :config.settings["devices"][self.board]["configuration"]["N_PIXELS"]]*r])
        # if there's a high (eg clap):
        if self.current_freq_detects["high"]:
            self.power_brightness = 1.0
            # Generate random indexes
            self.power_indexes = random.sample(range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"]), config.settings["devices"][self.board]["effect_opts"]["Power"]["s_count"])
            #print("ye")
        # Assign colour to the random indexes
        for index in self.power_indexes:
            output[0, index] = int(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Power"]["s_color"])[0]*self.power_brightness)
            output[1, index] = int(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Power"]["s_color"])[1]*self.power_brightness)
            output[2, index] = int(colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Power"]["s_color"])[2]*self.power_brightness)
        # Remove some of the indexes for next time
        self.power_indexes = [i for i in self.power_indexes if i not in random.sample(self.power_indexes, len(self.power_indexes)//4)]
        if len(self.power_indexes) <= 4:
            self.power_indexes = []
        # Fade the colour of the sparks out a bit for next time
        if self.power_brightness > 0:
            self.power_brightness -= 0.05
        # Calculate length of bass bar based on max bass frequency volume and length of strip
        strip_len = int((config.settings["devices"][self.board]["configuration"]["N_PIXELS"]//3)*max(y[:int(config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]*0.2)]))
        # Add the bass bars into the output. Colour proportional to length
        output[0][:strip_len] = colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][0][strip_len]
        output[1][:strip_len] = colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][1][strip_len]
        output[2][:strip_len] = colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Power"]["color_mode"]][2][strip_len]
        if config.settings["devices"][self.board]["effect_opts"]["Power"]["flip_lr"]:
            output = np.fliplr(output)
        if config.settings["devices"][self.board]["effect_opts"]["Power"]["mirror"]:
            output = np.concatenate((output[:, ::-2], output[:, ::2]), axis=1)
        return output

    def visualize_pulse(self, y):
        """dope ass visuals that's what"""
        config.settings["devices"][self.board]["effect_opts"]["Pulse"]["bar_color"]
        config.settings["devices"][self.board]["effect_opts"]["Pulse"]["bar_speed"]
        config.settings["devices"][self.board]["effect_opts"]["Pulse"]["bar_length"]
        config.settings["devices"][self.board]["effect_opts"]["Pulse"]["color_mode"]
        y = np.copy(interpolate(y, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        common_mode.update(y) # i honestly have no idea what this is but i just work with it rather than trying to figure it out
        self.prev_spectrum = np.copy(y)
        # Color channel mappings
        r = r_filt.update(y - common_mode.value) # same with this, no flippin clue
        r = np.array([j for i in zip(r,r) for j in i])
        output = np.array([colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Pulse"]["color_mode"]][0][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]],
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Pulse"]["color_mode"]][1][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]],
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Pulse"]["color_mode"]][2][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]]])
        
    def visualize_single(self):
        "Displays a single colour, non audio reactive"
        output = np.zeros((3,config.settings["devices"][self.board]["configuration"]["N_PIXELS"]))
        output[0][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Single"]["color"])[0]
        output[1][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Single"]["color"])[1]
        output[2][:]=colour_manager.colour(config.settings["devices"][self.board]["effect_opts"]["Single"]["color"])[2]
        return output

    def visualize_gradient(self):
        "Displays a multicolour gradient, non audio reactive"
        output = np.array([colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Gradient"]["color_mode"]][0][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]],
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Gradient"]["color_mode"]][1][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]],
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Gradient"]["color_mode"]][2][:config.settings["devices"][self.board]["configuration"]["N_PIXELS"]]])
        colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Gradient"]["color_mode"]] = np.roll(
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Gradient"]["color_mode"]],
                           config.settings["devices"][self.board]["effect_opts"]["Gradient"]["roll_speed"]*(-1 if config.settings["devices"][self.board]["effect_opts"]["Gradient"]["reverse"] else 1),
                           axis=1)
        if config.settings["devices"][self.board]["effect_opts"]["Gradient"]["mirror"]:
            output = np.concatenate((output[:, ::-2], output[:, ::2]), axis=1)
        return output

    def visualize_fade(self):
        "Fades through a multicolour gradient, non audio reactive"
        output = np.array([[colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Fade"]["color_mode"]][0][0] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])],
                           [colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Fade"]["color_mode"]][1][0] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])],
                           [colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Fade"]["color_mode"]][2][0] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])]])
        colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Fade"]["color_mode"]] = np.roll(
                           colour_manager.full_gradients[self.board][config.settings["devices"][self.board]["effect_opts"]["Fade"]["color_mode"]],
                           config.settings["devices"][self.board]["effect_opts"]["Fade"]["roll_speed"]*(-1 if config.settings["devices"][self.board]["effect_opts"]["Fade"]["reverse"] else 1),
                           axis=1)
        return output

    def visualize_calibration(self):
        "Custom values for RGB"
        output = np.array([[config.settings["devices"][self.board]["effect_opts"]["Calibration"]["r"] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])],
                           [config.settings["devices"][self.board]["effect_opts"]["Calibration"]["g"] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])],
                           [config.settings["devices"][self.board]["effect_opts"]["Calibration"]["b"] for i in range(config.settings["devices"][self.board]["configuration"]["N_PIXELS"])]])
        return output

class DSP(BoardManager):
    def __init__(self, board):
        # Name of board for which this dsp instance is processing audio
        self.board = board
        # Initialise filters etc
        self.fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]), alpha_decay=0.99, alpha_rise=0.99)
        self.mel_gain =        dsp.ExpFilter(np.tile(1e-1, config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]), alpha_decay=0.01, alpha_rise=0.99)
        self.mel_smoothing =   dsp.ExpFilter(np.tile(1e-1, config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]), alpha_decay=0.2, alpha_rise=0.99)
        self.gain =            dsp.ExpFilter(np.tile(0.01, config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"]), alpha_decay=0.001, alpha_rise=0.99)
        self.r_filt =          dsp.ExpFilter(np.tile(0.01, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2), alpha_decay=0.2, alpha_rise=0.99)
        self.g_filt =          dsp.ExpFilter(np.tile(0.01, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2), alpha_decay=0.05, alpha_rise=0.3)
        self.b_filt =          dsp.ExpFilter(np.tile(0.01, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2), alpha_decay=0.1, alpha_rise=0.5)
        self.common_mode =     dsp.ExpFilter(np.tile(0.01, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2), alpha_decay=0.99, alpha_rise=0.01)
        self.p_filt =          dsp.ExpFilter(np.tile(1, (3, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2)), alpha_decay=0.1, alpha_rise=0.99)
        self.volume =          dsp.ExpFilter(config.settings["configuration"]["MIN_VOLUME_THRESHOLD"], alpha_decay=0.9, alpha_rise=0.02)
        self.p =               np.tile(1.0, (3, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] // 2))
        # Number of audio samples to read every time frame
        self.samples_per_frame = int(config.settings["mic_config"]["MIC_RATE"] / config.settings["configuration"]["FPS"])
        # Array containing the rolling audio sample window
        self.y_roll = np.random.rand(config.settings["configuration"]["N_ROLLING_HISTORY"], self.samples_per_frame) / 1e16
        self.fft_window =      np.hamming(int(config.settings["mic_config"]["MIC_RATE"] / config.settings["configuration"]["FPS"])\
                                         * config.settings["configuration"]["N_ROLLING_HISTORY"])

        self.samples = None
        self.mel_y = None
        self.mel_x = None
        self.create_mel_bank()

    def update(self, audio_samples):
        """ Return processed audio data
        Returns mel curve, x/y data
        This is called every time there is a microphone update
        Returns
        -------
        audio_data : dict
            Dict containinng "mel", "vol", "x", and "y"
        """
        audio_data = {}
        # Normalize samples between 0 and 1
        y = audio_samples / 2.0**15
        # Construct a rolling window of audio samples
        self.y_roll[:-1] = self.y_roll[1:]
        self.y_roll[-1, :] = np.copy(y)
        y_data = np.concatenate(self.y_roll, axis=0).astype(np.float32)
        vol = np.max(np.abs(y_data))
        # Transform audio input into the frequency domain
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= self.fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * self.mel_y.T
        # Scale data to values more suitable for visualization
        mel = np.sum(mel, axis=0)
        mel = mel**2
        # Gain normalization
        self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=0.1)))
        mel /= self.mel_gain.value
        mel = self.mel_smoothing.update(mel)
        x = np.linspace(config.settings["devices"][self.board]["configuration"]["MIN_FREQUENCY"], config.settings["devices"][self.board]["configuration"]["MAX_FREQUENCY"], len(mel))
        y = self.fft_plot_filter.update(mel)

        audio_data["mel"] = mel
        audio_data["vol"] = vol
        audio_data["x"]   = x
        audio_data["y"]   = y
        return audio_data

    def rfft(self, data, window=None):
        window = 1.0 if window is None else window(len(data))
        ys = np.abs(np.fft.rfft(data * window))
        xs = np.fft.rfftfreq(len(data), 1.0 / config.settings["mic_config"]["MIC_RATE"])
        return xs, ys

    def fft(self, data, window=None):
        window = 1.0 if window is None else window(len(data))
        ys = np.fft.fft(data * window)
        xs = np.fft.fftfreq(len(data), 1.0 / config.settings["mic_config"]["MIC_RATE"])
        return xs, ys

    def create_mel_bank(self):
        samples = int(config.settings["mic_config"]["MIC_RATE"] * config.settings["configuration"]["N_ROLLING_HISTORY"]\
                                                   / (2.0 * config.settings["configuration"]["FPS"]))
        self.mel_y, (_, self.mel_x) = melbank.compute_melmat(num_mel_bands=config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"],
                                                             freq_min=config.settings["devices"][self.board]["configuration"]["MIN_FREQUENCY"],
                                                             freq_max=config.settings["devices"][self.board]["configuration"]["MAX_FREQUENCY"],
                                                             num_fft_bands=samples,
                                                             sample_rate=config.settings["mic_config"]["MIC_RATE"])

class ColourManager():
    """
    Controls all colours and gradients, both user set and default
    Colours and gradients are stored in two dicts, one for defaults and one for user set
    Format for colours:   "Red":(255,0,0)
    Format for gradients: "Ocean":[(0, 255, 0), (0, 247, 161), (0, 0, 255)]
    """
    def __init__(self):
        self.colours_storage = QSettings('./lib/colours.ini', QSettings.IniFormat)
        self.gradients_storage = QSettings('./lib/gradients.ini', QSettings.IniFormat)
        self.colours_storage.setFallbacksEnabled(False)
        self.gradients_storage.setFallbacksEnabled(False)
        self.effects_using_colours = []
        self.effects_using_gradients = []
        # load user set colours and gradients
        self.loadFromINI(self.colours_storage, "colours")
        self.loadFromINI(self.gradients_storage, "gradients")
        # default colours and gradients
        self.loadDefaultColours()
        self.loadDefaultGradients()
        self.buildGradients()
        # find and save which effects have settings for colours or gradients
        for board in config.settings["devices"]:
            for effect in config.dynamic_effects_config:
                for setting in config.dynamic_effects_config[effect]:
                    if setting[2] == "dropdown":
                        if setting[3] == "colours":
                            self.effects_using_colours.append((board, effect, setting[0]))
                        elif setting[3] == "gradients":
                            self.effects_using_gradients.append((board, effect, setting[0]))

    def buildGradients(self):
        # generate full gradients for each device
        self.full_gradients = {}
        for board in config.settings["devices"]:
            self.full_gradients[board] = {}
            for i in ["user", "default"]:
                for gradient in config.colour_manager[i+"_gradients"]:
                    self.full_gradients[board][gradient] = self._easing_gradient_generator(config.colour_manager[i+"_gradients"][gradient],
                                                                                           config.settings["devices"][board]["configuration"]["N_PIXELS"])
                    self.full_gradients[board][gradient] = np.concatenate((self.full_gradients[board][gradient][:, ::-1],
                                                                           self.full_gradients[board][gradient]), axis=1)

    def colour(self, colour):
        # returns the values of a given colour. use this function to get colour values.
        if colour in config.colour_manager["user_colours"]:
            return config.colour_manager["user_colours"][colour]
        elif colour in config.colour_manager["default_colours"]:
            return config.colour_manager["default_colours"][colour]
        else:
            print("colour {} has not been defined".format(colour))
            return (0,0,0)

    def getColours(self, group):
        # "user" returns user colours
        # "default" returns default colours
        # "all" returns a list of both
        if group == "user":
            return config.colour_manager["user_colours"]
        elif group == "default":
            return config.colour_manager["default_colours"]
        elif group == "all":
            return {**config.colour_manager["user_colours"], **config.colour_manager["default_colours"]}

    def getGradients(self, group):
        if group == "user":
            return config.colour_manager["user_gradients"]
        elif group == "default":
            return config.colour_manager["default_gradients"]
        elif group == "all":
            return {**config.colour_manager["user_gradients"], **config.colour_manager["default_gradients"]}

    def _easing_gradient_generator(self, colors, length):
        """
        returns np.array of given length that eases between specified colours

        parameters:
        colors - list, colours must be in config.colour_manager["colours"]
            eg. ["Red", "Orange", "Blue", "Purple"]
        length - int, length of array to return. should be from config.settings
            eg. config.settings["devices"]["my strip"]["configuration"]["N_PIXELS"]
        """
        def _easing_func(x, length, slope=2.5):
            # returns a nice eased curve with defined length and curve
            xa = (x/length)**slope
            return xa / (xa + (1 - (x/length))**slope)
        colors = colors[::-1] # needs to be reversed, makes it easier to deal with
        n_transitions = len(colors) - 1
        ease_length = length // n_transitions
        pad = length - (n_transitions * ease_length)
        output = np.zeros((3, length))
        ease = np.array([_easing_func(i, ease_length, slope=2.5) for i in range(ease_length)])
        # for r,g,b
        for i in range(3):
            # for each transition
            for j in range(n_transitions):
                # Starting ease value
                start_value = colors[j][i]
                # Ending ease value
                end_value = colors[j+1][i]
                # Difference between start and end
                diff = end_value - start_value
                # Make array of all start value
                base = np.empty(ease_length)
                base.fill(start_value)
                # Make array of the difference between start and end
                diffs = np.empty(ease_length)
                diffs.fill(diff)
                # run diffs through easing function to make smooth curve
                eased_diffs = diffs * ease
                # add transition to base values to produce curve from start to end value
                base += eased_diffs
                # append this to the output array
                output[i, j*ease_length:(j+1)*ease_length] = base
        # cast to int
        output = np.asarray(output, dtype=int)
        # pad out the ends (bit messy but it works and looks good)
        if pad:
            for i in range(3):
                output[i, -pad:] = output[i, -pad-1]
        return output

    def loadFromINI(self, settings, toLoad, overwrite=True):
        # loads colours or gradients from INI save file
        # toLoad: "colours" or "gradients"
        # settings is the QSettings object
        # overwrite kwarg determines if duplicates are overwritten from file or not.
        storage = settings.value(toLoad)
        if storage:
            try:
                config.colour_manager["user_"+toLoad] = {**config.colour_manager["user_"+toLoad], **settings.value(toLoad)} if overwrite else\
                                                {**settings.value(toLoad), **config.colour_manager["user_"+toLoad]}
            except TypeError:
                print("Error parsing {} from file: {}".format(toLoad, storage))
                pass
        else:
            print("No user {} found".format(toLoad))

    def addColour(self, group, colour_name, colour_value):
        # can be used to add a new colour, or modify an existing value
        assert(group in ["user", "default"]), "invalic group: {}".format(group)
        assert(colour_name not in self.getColours("all")), "Colour {} already exists".format(colour_name)
        assert(type(colour_value) == tuple), "Colour_value {} not tuple format".format(colour_value)
        config.colour_manager[group+"_colours"][colour_name] = colour_value

    def editColour(self, group, old_name, colour_value, new_name=None):
        # changes the name and/or value of a colour
        assert(group in ["user", "default"]), "Invalic group: {}".format(group)
        assert(old_name in self.getColours("all")), "Colour {} does not exist".format(old_name)
        assert(type(colour_value) == tuple), "Colour_value {} not tuple format".format(colour_value)
        if group == "default":
            assert(not new_name), "Not allowed to edit default colour names"
        if new_name:
            self.delColour(group, old_name)
            old_name = new_name
        config.colour_manager[group+"_colours"][old_name] = colour_value

    def delColour(self, group, colour_name):
        # delete a saved colour.
        assert(group in ["user", "default"]), "Invalic group: {}".format(group)
        assert(group in ["user"]), "Deleting default colours not allowed"
        del config.colour_manager[group+"_colours"][colour_name]
        self.removeReferences("colour", colour_name)

    def addGradient(self, gradient_name, gradient_colours):
        # can be used to add a new gradient, or modify an existing one
        config.colour_manager["user_gradients"][gradient_name] = gradient_colours

    def delGradient(self, gradient_name):
        # delete a saved gradient
        del config.colour_manager["user_gradients"][gradient_name]

    def loadDefaultColours(self):
        # Loads default colours.
        config.colour_manager["default_colours"] = config.default_colours.copy()

    def loadDefaultGradients(self):
        # Loads default gradients.
        config.colour_manager["default_gradients"] = config.default_gradients

    def saveColours(self):
        self.colours_storage.setValue("colours", config.colour_manager["user_colours"])
        self.colours_storage.sync()

    def saveGradients(self):
        self.gradients_storage.setValue("gradients", config.colour_manager["user_gradients"])
        self.gradients_storage.sync()

    def removeReferences(self, to_remove, name, new_name=None):
        # cleans settings of a colour/gradient
        # eg if "red" is deleted, all usages of "red" will be changed to "black"
        # to_remove: "colour" or "gradient"
        # name: name of thing to remove references of
        # new_name: optional, change name to something else
        null_colour = "Black"
        null_gradient = "Spectral"
        if to_remove == "colour":
            if new_name in colour_manager.getColours("all"):
                null_colour = new_name
        elif to_remove == "gradient":
            if new_name in colour_manager.getGradients("all"):
                null_gradient = new_name
        if to_remove == "colour":
            for effect in self.effects_using_colours:
                if config.settings["devices"][effect[0]]["effect_opts"][effect[1]][effect[2]] == name:
                    print(effect, "set to", null_colour)
                    config.settings["devices"][effect[0]]["effect_opts"][effect[1]][effect[2]] = null_colour
        elif to_remove == "gradient":
            for effect in self.effects_using_gradients:
                if config.settings["devices"][effect[0]]["effect_opts"][effect[1]][effect[2]] == name:
                    config.settings["devices"][effect[0]]["effect_opts"][effect[1]][effect[2]] = null_gradient

class Microphone():
    """Controls the audio input, allowing device selection and streaming"""
    def __init__(self, callback_func):
        # in this class, "device" is used to refer to the audio device, not "device" in the context of the led strips
        self.callback_func = callback_func
        self.numdevices = py_audio.get_device_count()
        self.default_device_id = py_audio.get_default_input_device_info()['index']
        self.devices = []

        #for each audio device, add to list of devices
        for i in range(0,self.numdevices):
            device_info = py_audio.get_device_info_by_host_api_device_index(0,i)
            if device_info["maxInputChannels"] > 1:
                self.devices.append(device_info)

        if not "MIC_ID" in config.settings["mic_config"]:
            self.setDevice(self.default_device_id)
        else:
            self.setDevice(config.settings["mic_config"]["MIC_ID"])

    def getDevices(self):
        return self.devices

    def setDevice(self, device_id):
        # set device to stream from by the id of the device
        if not device_id in range(0,self.numdevices):
            raise ValueError("No device with id {}".format(device_id))
        self.device_id = self.devices[device_id]["index"]
        self.device_name = self.devices[device_id]["name"]
        self.device_rate = int(self.devices[device_id]["defaultSampleRate"])
        self.frames_per_buffer = self.device_rate // config.settings["configuration"]["FPS"]

        config.settings["mic_config"]["MIC_ID"] = self.device_id
        config.settings["mic_config"]["MIC_NAME"] = self.device_name
        config.settings["mic_config"]["MIC_RATE"] = self.device_rate

    def startStream(self):
        self.stream = py_audio.open(format = pyaudio.paInt16,
                                    channels = 1,
                                    rate = self.device_rate,
                                    input = True,
                                    input_device_index = self.device_id,
                                    frames_per_buffer = self.frames_per_buffer)
            
        # overflows = 0
        # prev_ovf_time = time.time()
        while True:
            try:
                y = np.fromstring(self.stream.read(self.frames_per_buffer), dtype=np.int16)
                y = y.astype(np.float32)
                self.callback_func(y)
            except IOError:
                pass
                # overflows += 1
                # if time.time() > prev_ovf_time + 1:
                #     prev_ovf_time = time.time()
                #     if config.settings["configuration"]["USE_GUI"]:
                #         gui.label_error.setText('Audio buffer has overflowed {} times'.format(overflows))
                #     else:
                #         print('Audio buffer has overflowed {} times'.format(overflows))

    def stopStream(self):
        self.stream.stop_stream()
        self.stream.close()

class GUI(QMainWindow):
    """The graphical interface of the application"""
    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.updateUIVisibleItems()

    def initMainWindow(self):
        # Set up window and wrapping layout
        self.setWindowTitle("Visualization")
        # Initial window size/pos last saved if available
        settings.beginGroup("MainWindow")
        if settings.value("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        else:
            self.setGeometry(100,100,500,300)
        if settings.value("state"):
            self.restoreState(settings.value("state"))
        settings.endGroup()
        self.main_wrapper = QVBoxLayout()

        # Set up toolbar
        #toolbar_guiDialogue.setShortcut('Ctrl+H')
        toolbar_deviceDialogue = QAction('LED Strip Manager', self)
        toolbar_deviceDialogue.triggered.connect(self.deviceDialogue)
        toolbar_micDialogue = QAction('Microphone Setup', self)
        toolbar_micDialogue.triggered.connect(self.micDialogue)
        toolbar_colourDialogue = QAction('Colour Control', self)
        toolbar_colourDialogue.triggered.connect(self.colourDialogue)
        toolbar_guiDialogue = QAction('GUI Properties', self)
        toolbar_guiDialogue.triggered.connect(self.guiDialogue)
        toolbar_saveDialogue = QAction('Save Settings', self)
        toolbar_saveDialogue.triggered.connect(self.saveDialogue)
        
        self.toolbar = self.addToolBar('top_toolbar')
        self.toolbar.setObjectName('top_toolbar')
        self.toolbar.addAction(toolbar_deviceDialogue)
        self.toolbar.addAction(toolbar_micDialogue)
        self.toolbar.addAction(toolbar_colourDialogue)
        self.toolbar.addAction(toolbar_guiDialogue)
        self.toolbar.addAction(toolbar_saveDialogue)

        # Set up FPS and error labels
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.label_error = QLabel("")
        self.label_fps = QLabel("")
        self.label_latency = QLabel("")
        self.label_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label_latency.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusbar.addPermanentWidget(self.label_error, stretch=1)
        self.statusbar.addPermanentWidget(self.label_latency)
        self.statusbar.addPermanentWidget(self.label_fps)

        # Set up board tabs widget
        self.label_boards = QLabel("LED Strips")
        self.boardsTabWidget = QTabWidget()
        # Dynamically set up boards tabs
        self.gui_widgets = {}        # contains references to areas of gui for visibility settings
        self.gui_widgets["Graphs"] = []
        self.gui_widgets["Reactive Effect Buttons"] = []
        self.gui_widgets["Non Reactive Effect Buttons"] = []
        self.gui_widgets["Frequency Range"] = []
        self.gui_widgets["Effect Options"] = []
        self.board_tabs = {}         # contains all the tabs, one for each board
        self.board_tabs_widgets = {} # contains all the widgets for each tab

        self.main_wrapper.addWidget(self.label_boards)
        self.main_wrapper.addWidget(self.boardsTabWidget)
        #self.setLayout(self.main_wrapper)

        # Set up setupHelper
        self.initSetupHelper()

        # Set wrapper as main widget
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(self.main_wrapper)
        self.show()

    def initSetupHelper(self):
        helpstring = """
Looks like you need to connect an LED strip!\n\n
1: Open the 'LED Strip Manager' on the toolbar above\n
2: Choose the type of device (eg ESP8266) from the dropdown menu\n
3: Fill in the boxes with connection and settings info\n
4: Add the device\n
5: Party time!\n\n
If you have any questions, feel free to open an issue on the GitHub page.
"""
        self.setupHelper = QWidget()
        self.setupHelperLayout = QVBoxLayout()
        self.setupHelper.setLayout(self.setupHelperLayout)
        self.setupHelperText = QLabel(helpstring)
        self.setupHelperText.setWordWrap(True)
        self.setupHelperLayout.addWidget(self.setupHelperText)
        self.setupHelperLayout.addStretch()

    def showSetupHelper(self):
        self.boardsTabWidget.addTab(self.setupHelper, "Setup Helper")

    def hideSetupHelper(self):
        idx = self.boardsTabWidget.indexOf(self.setupHelper)
        self.boardsTabWidget.removeTab(idx)

    def addBoard(self, board):
        self.board_tabs_widgets[board] = {}
        self.board_tabs[board] = QWidget()

        self.initBoardUI(board)
        self.boardsTabWidget.addTab(self.board_tabs[board],board)
        self.board_tabs[board].setLayout(self.board_tabs_widgets[board]["wrapper"])

        self.gui_widgets["Graphs"].append([self.board_tabs_widgets[board]["graph_view"]])
        self.gui_widgets["Reactive Effect Buttons"].append([self.board_tabs_widgets[board]["label_reactive"], self.board_tabs_widgets[board]["reactive_button_grid_wrap"]])
        self.gui_widgets["Non Reactive Effect Buttons"].append([self.board_tabs_widgets[board]["label_non_reactive"], self.board_tabs_widgets[board]["non_reactive_button_grid_wrap"]])
        self.gui_widgets["Frequency Range"].append([self.board_tabs_widgets[board]["label_slider"], self.board_tabs_widgets[board]["freq_slider"]])
        self.gui_widgets["Effect Options"].append([self.board_tabs_widgets[board]["label_options"], self.board_tabs_widgets[board]["opts_tabs"]])
        self.updateUIVisibleItems()


    def delBoard(self, board):
        idx = self.boardsTabWidget.indexOf(self.board_tabs[board])
        self.boardsTabWidget.removeTab(idx)
        #self.gui_widgets["Graphs"].remove([self.board_tabs_widgets[board]["graph_view"]])
        #self.gui_widgets["Reactive Effect Buttons"].remove([self.board_tabs_widgets[board]["label_reactive"], self.board_tabs_widgets[board]["reactive_button_grid_wrap"]])
        #self.gui_widgets["Non Reactive Effect Buttons"].remove([self.board_tabs_widgets[board]["label_non_reactive"], self.board_tabs_widgets[board]["non_reactive_button_grid_wrap"]])
        #self.gui_widgets["Frequency Range"].remove([self.board_tabs_widgets[board]["label_slider"], self.board_tabs_widgets[board]["freq_slider"]])
        #self.gui_widgets["Effect Options"].remove([self.board_tabs_widgets[board]["label_options"], self.board_tabs_widgets[board]["opts_tabs"]])
        #del self.board_tabs_widgets[board]
        self.board_tabs[board].deleteLater()
        self.updateUIVisibleItems()

    def closeEvent(self, event):
        # executed when the window is being closed
        quit_msg = "Are you sure you want to exit?"
        reply = QMessageBox.question(self, 'Exit', 
                         quit_msg, QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Save window state
            settings.beginGroup("MainWindow")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue('state', self.saveState())
            settings.endGroup()
            # save all settings
            settings.setValue("settings_dict", config.settings)
            # save and close
            settings.sync()
            colour_manager.saveColours()
            colour_manager.saveGradients()
            event.accept()
            sys.exit(0)
            
        else:
            event.ignore()

    def updateUIVisibleItems(self):
        for section in self.gui_widgets:
            for widgets in self.gui_widgets[section]:
                for widget in widgets:
                    widget.setVisible(config.settings["GUI_opts"][section])
        if len(config.settings["devices"]) == 0:
            self.showSetupHelper()
        else:
            self.hideSetupHelper()

    def colourDialogue(self):
        import re

        def cleanColourGrid():
            # cleanup old colour widgets referring to ones that no longer exist
            for widget in self.colourDialogueWidgets:
                widget.deleteLater()

        def updateColourGrid():
            def delFuncGenerator(name, group):
                def func():
                    colour_manager.delColour(group, name)
                    cleanColourGrid()
                    gui.updateColourDropdowns()
                    updateColourGrid()
                func.__name__ = name
                return func

            def editFuncGenerator(name, group):
                def func():
                    value = colour_manager.colour(name)
                    addColourDialogue(group=group, name=name, value=value, edit=True)
                    cleanColourGrid()
                    gui.updateColourDropdowns()
                    updateColourGrid()
                func.__name__ = name
                return func

            self.colourDialogueWidgets = []
            self.colourDialogueGridEditFuncs = {}
            self.colourDialogueGridDeleteFuncs = {}
            del_enabled = {"user": True, "default": False}
            
            for i in ["user", "default"]:
                colours = colour_manager.getColours(i)
                count = 0
                for colour in colours:
                    self.colourDialogueGridEditFuncs[colour] = editFuncGenerator(colour, i)
                    self.colourDialogueGridDeleteFuncs[colour] = delFuncGenerator(colour, i)
                    label = QLabel(colour)
                    colourBox = QWidget()
                    delButton = QPushButton("")
                    delButton.setIcon(QIcon('./lib/bin.png'))
                    delButton.setIconSize(QSize(14,14))
                    delButton.clicked.connect(self.colourDialogueGridDeleteFuncs[colour])
                    delButton.setEnabled(del_enabled[i])
                    editButton = QPushButton("")
                    editButton.setIcon(QIcon('./lib/edit.png'))
                    editButton.clicked.connect(self.colourDialogueGridEditFuncs[colour])
                    editButton.setIconSize(QSize(14,14))
                    p = colourBox.palette()
                    p.setColor(colourBox.backgroundRole(), QColor(*colours[colour]))
                    colourBox.setPalette(p)
                    colourBox.setAutoFillBackground(True)
                    self.pallettes[i]["grid"].setColumnStretch(0,1)
                    self.pallettes[i]["grid"].setColumnStretch(1,1)
                    self.pallettes[i]["grid"].addWidget(label,count,0)
                    self.pallettes[i]["grid"].addWidget(colourBox,count,1)
                    self.pallettes[i]["grid"].addWidget(editButton,count,2)
                    self.pallettes[i]["grid"].addWidget(delButton,count,3)
                    self.colourDialogueWidgets.extend([label, colourBox, editButton, delButton])
                    count += 1

        def restoreDefaultColours():
            colour_manager.loadDefaultColours()
            cleanColourGrid()
            gui.updateColourDropdowns()
            updateColourGrid()

        def addColourDialogue(name=None, value=None, edit=False, group="user"):
            self.initial_value = value
            def addColour():
                rgb = self.addColourPreviewBoxPalette.color(self.addColourPreviewBox.backgroundRole()).getRgb()
                if edit:
                    colour_manager.editColour(group, name, rgb, new_name=(self.addColourEditName.text() if group=="user" else None))
                else:
                    colour_manager.addColour(group, self.addColourEditName.text(), rgb)
                cleanColourGrid()
                gui.updateColourDropdowns()
                updateColourGrid()
                self.addColourDialogue.accept()

            def cancelChanges():
                setPreviewColour(QColor(*self.initial_value if self.initial_value else (255,255,255)))

            def acceptChanges():
                self.initial_value = self.addColourPreviewBoxPalette.color(self.addColourPreviewBox.backgroundRole()).getRgb()
                setPreviewColour(QColor(*self.initial_value if self.initial_value else (255,255,255)))

            def showColourDialogue():
                self.colourEditDialogue = QColorDialog(QColor(*self.initial_value if self.initial_value else (255,255,255)))
                self.colourEditDialogue.setWindowModality(Qt.ApplicationModal)
                self.colourEditDialogue.show()
                self.colourEditDialogue.accepted.connect(acceptChanges)
                self.colourEditDialogue.rejected.connect(cancelChanges)
                self.colourEditDialogue.currentColorChanged.connect(setPreviewColour)

            def setPreviewColour(colour):
                self.addColourPreviewBoxPalette.setColor(self.addColourPreviewBox.backgroundRole(), colour)
                self.addColourPreviewBox.setPalette(self.addColourPreviewBoxPalette)
                if edit:
                    colour_manager.editColour(group, name, colour.getRgb(), new_name=(self.addColourEditName.text() if group=="user" else None))

            def validColourName(name):
                styles = ["", "border: 1px solid #3be820;", "border: 1px solid red;"]
                if name:
                    if re.match("\w+$", name):
                        self.addColourEditName.setStyleSheet(styles[1])
                        buttons.button(QDialogButtonBox.Ok).setEnabled(True)
                    else:
                        self.addColourEditName.setStyleSheet(styles[2])
                        buttons.button(QDialogButtonBox.Ok).setEnabled(False)
                else:
                    self.addColourEditName.setStyleSheet(styles[0])
                    buttons.button(QDialogButtonBox.Ok).setEnabled(False)

            # Set up window and layout
            self.addColourDialogue = QDialog(None, Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
            self.addColourDialogue.setWindowTitle("{} Colour".format("Edit" if edit else "Add"))
            self.addColourDialogue.setWindowModality(Qt.ApplicationModal)
            layout = QGridLayout()
            self.addColourDialogue.setLayout(layout)
    
            # Colour name input
            self.addColourLabelName = QLabel("Name")
            self.addColourEditName = QLineEdit()
            self.addColourEditName.setEnabled(True if group == "user" else False)
            self.addColourEditName.textChanged.connect(validColourName)
            layout.addWidget(self.addColourLabelName, 0, 0)
            layout.addWidget(self.addColourEditName, 0, 1, 1, 2)

            # Colour value input
            self.addColourLabelColour = QLabel("Colour")
            self.addColourPreviewBox = QWidget()
            self.addColourPreviewBoxPalette = self.addColourPreviewBox.palette()
            self.addColourPreviewBoxPalette.setColor(self.addColourPreviewBox.backgroundRole(), QColor(*self.initial_value if self.initial_value else (255,255,255)))
            self.addColourPreviewBox.setPalette(self.addColourPreviewBoxPalette)
            self.addColourPreviewBox.setAutoFillBackground(True)
            self.colourEdit = QPushButton("Edit Colour")
            self.colourEdit.clicked.connect(showColourDialogue)
            layout.addWidget(self.addColourLabelColour, 1, 0)
            layout.addWidget(self.addColourPreviewBox, 1, 1)
            layout.addWidget(self.colourEdit, 1, 2)
    
            # Set up dialogue buttons
            buttons = QDialogButtonBox(Qt.Horizontal, self)
            buttons.addButton(QDialogButtonBox.Ok)
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            buttons.accepted.connect(addColour)
            layout.addWidget(buttons, 2, 0, 1, 3)
            layout.setColumnStretch(1,1)
            if name:
                self.addColourEditName.setText(name)
            self.addColourDialogue.show()

        # Set up window and layout
        self.colour_dialogue = QDialog(None, Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        self.colour_dialogue.setWindowTitle("Colour Control")
        self.colour_dialogue.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout()
        self.colour_dialogue.setLayout(layout)

        # Set up colour display
        self.pallettes = {}
        self.pallettes["user"] = {}
        self.pallettes["default"] = {}
        self.pallettes["buttons"] = {}
        for i in ["user", "default"]:
            self.pallettes[i]["groupbox"] = QGroupBox()
            self.pallettes[i]["wrapper_layout"] = QVBoxLayout()
            self.pallettes[i]["grid"] = QGridLayout()
            self.pallettes[i]["groupbox"].setLayout(self.pallettes[i]["wrapper_layout"])
            self.pallettes[i]["wrapper_layout"].addLayout(self.pallettes[i]["grid"], stretch=1)
        restore = QPushButton("Reset Defaults")
        add = QPushButton("Add Colour")
        restore.clicked.connect(restoreDefaultColours)
        add.clicked.connect(addColourDialogue)
        self.pallettes["buttons"]["wrap"] = QWidget()
        self.pallettes["buttons"]["layout"] = QHBoxLayout()
        self.pallettes["buttons"]["layout"].addStretch()
        self.pallettes["buttons"]["layout"].addWidget(restore)
        self.pallettes["buttons"]["layout"].addWidget(add)
        self.pallettes["buttons"]["wrap"].setLayout(self.pallettes["buttons"]["layout"])
        self.pallettes["user"]["groupbox"].setTitle("Custom colours")
        self.pallettes["default"]["groupbox"].setTitle("Default colours")
        layout.addWidget(self.pallettes["default"]["groupbox"])
        layout.addWidget(self.pallettes["user"]["groupbox"])
        layout.addWidget(self.pallettes["buttons"]["wrap"])

        # Load and show colours
        updateColourGrid()

        # Set up dialogue buttons
        self.buttons = QDialogButtonBox(Qt.Horizontal, self)
        self.buttons.addButton(QDialogButtonBox.Ok)
        self.buttons.accepted.connect(self.colour_dialogue.accept)
        layout.addWidget(self.buttons)
        self.colour_dialogue.show()

    def micDialogue(self):
        def set_mic():
            microphone.setDevice(mic_button_group.checkedId())
            microphone.startStream()

        # Set up window and layout
        self.mic_dialogue = QDialog(None, Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        self.mic_dialogue.setWindowTitle("Mic Setup")
        self.mic_dialogue.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout()
        self.mic_dialogue.setLayout(layout)

        # Set up buttons for each mic
        mic_button_group = QButtonGroup(self.mic_dialogue)
        mic_buttons = {}
        mics = microphone.getDevices()
        for mic in mics:
            mic_id = mic["index"]
            mic_buttons[mic_id] = QRadioButton(mic["name"])
            mic_button_group.addButton(mic_buttons[mic_id])
            mic_button_group.setId(mic_buttons[mic_id], mic_id)
            mic_buttons[mic_id].clicked.connect(set_mic)
            if config.settings["mic_config"]["MIC_ID"] == mic_id:
                mic_buttons[mic_id].setChecked(Qt.Checked)
            layout.addWidget(mic_buttons[mic_id])

        # Set up ok/cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.mic_dialogue.accept)
        layout.addWidget(self.buttons)
        self.mic_dialogue.show()

    def deviceDialogue(self):
        def addBoard_to_manager():
            general_config = {}
            required_config = {}
            current_device = device_type_cbox.currentText()
            for gen_config_setting in config.device_gen_config:
                if config.device_gen_config[gen_config_setting][2] == "textbox":
                    general_config[gen_config_setting] = gen_widgets[gen_config_setting][1].text()
                elif config.device_gen_config[gen_config_setting][2] == "textbox-int":
                    general_config[gen_config_setting] = int(gen_widgets[gen_config_setting][1].text())
                elif config.device_gen_config[gen_config_setting][2] == "checkbox":
                    general_config[gen_config_setting] = gen_widgets[gen_config_setting][1].isChecked()
            if config.device_req_config[current_device]:
                for req_config_setting in config.device_req_config[current_device]:
                    if config.device_req_config[current_device][req_config_setting][2] == "textbox":
                        required_config[req_config_setting] = req_widgets[current_device][req_config_setting][1].text()
                    elif config.device_req_config[current_device][req_config_setting][2] == "textbox-int":
                        required_config[req_config_setting] = int(req_widgets[current_device][req_config_setting][1].text())
                    elif config.device_req_config[current_device][req_config_setting][2] == "checkbox":
                        required_config[req_config_setting] = req_widgets[current_device][req_config_setting][1].isChecked()
            required_config["TYPE"] = current_device
            board_name = general_config["NAME"]
            del general_config["NAME"]
            board_manager.addBoard(board_name, config_exists=False,
                                               req_config=required_config,
                                               gen_config=general_config)

        def remBoard_from_manager():
            for board in self.to_delete:
                board_manager.delBoard(board)
            for widget in self.del_widgets:
                self.del_widgets[widget][0].deleteLater()
                self.del_widgets[widget][1].deleteLater()
            populate_remove_device_list()

        def populate_remove_device_list():
            self.del_widgets = {}
            i = 0
            for board in config.settings["devices"]:
                # Make widgets
                wLabel = QLabel(board)
                wEdit = QCheckBox()
                wEdit.setCheckState(Qt.Unchecked)
                wEdit.stateChanged.connect(validate_rem_device_checks)
                self.del_widgets[board] = [wLabel, wEdit]
                # Add to layout
                remDeviceTabButtonLayout.addWidget(self.del_widgets[board][0], i, 0)
                remDeviceTabButtonLayout.addWidget(self.del_widgets[board][1], i, 1)
                i += 1

        def validate_rem_device_checks():
            self.to_delete = []
            for board in config.settings["devices"]:
                if self.del_widgets[board][1].isChecked():
                    self.to_delete.append(board)
            self.rem_device_button.setEnabled(True if self.to_delete else False)


        def show_hide_addBoard_interface():
            current_device = device_type_cbox.currentText()
            for device in config.device_req_config:
                for req_config_setting in req_widgets[device]:
                    if req_config_setting is not "no_config":
                        for widget in req_widgets[device][req_config_setting]:
                            widget.setVisible(device == current_device)
                    else:
                        req_widgets[device][req_config_setting].setVisible(device == current_device)

        def validate_inputs():
            # Checks all inputs are ok, before setting "add device" to usable
            import re
            current_device = device_type_cbox.currentText()
            req_valid_inputs = {}
            gen_valid_inputs = {}
            styles = ["", "border: 1px solid #3be820;", "border: 1px solid red;"]
            def valid_mac(x):
                return True if re.match("[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", test.lower()) else False
            def valid_ip(x):
                try:
                    pieces = x.split('.')
                    if len(pieces) != 4: return False
                    return all(0<=int(i)<256 for i in pieces)
                except:
                    return False
            def valid_int(x):
                try:
                    x = int(x)
                    return x > 0
                except:
                    return False
            def validate_address_port(x):
                try:
                    pieces = x.split(":")
                    if len(pieces) != 2: return False
                    return (((pieces[0] == "localhost") or (valid_ip(pieces[0]))) and valid_int(pieces[1])) is True
                except:
                    return False
            def update_req_box_highlight(setting, val):
                req_widgets[current_device][setting][1].setStyleSheet(styles[val])
            def update_gen_box_highlight(setting, val):
                gen_widgets[setting][1].setStyleSheet(styles[val])
            def update_add_device_button():
                req_value = all(req_valid_inputs[setting] for setting in config.device_req_config[current_device]) if config.device_req_config[current_device] else True
                gen_value = all(gen_valid_inputs[setting] for setting in config.device_gen_config) if config.device_gen_config else True
                self.add_device_button.setEnabled(req_value and gen_value)
            # Validate the inputs, highlight invalid boxes
            # ESP8266
            if current_device == "ESP8266":
                for req_config_setting in config.device_req_config[current_device]:
                    test = req_widgets[current_device][req_config_setting][1].text()
                    # Validate MAC
                    if req_config_setting == "MAC_ADDR":
                        valid = valid_mac(test)
                        req_widgets[current_device][req_config_setting][1].setText(test.replace(":", "-"))
                    # Validate IP
                    elif req_config_setting == "UDP_IP":
                        valid = valid_ip(test)
                    # Validate port
                    elif req_config_setting == "UDP_PORT":
                        valid = valid_int(test)
                    else:
                        valid = True
                    req_valid_inputs[req_config_setting] = valid
                    update_req_box_highlight(req_config_setting, (1 if valid else 2) if test else 0)
            # PxMatrix
            if current_device == "PxMatrix":
                for req_config_setting in config.device_req_config[current_device]:
                    test = req_widgets[current_device][req_config_setting][1].text()
                    # Validate MAC
                    if req_config_setting == "MAC_ADDR":
                        valid = valid_mac(test)
                        req_widgets[current_device][req_config_setting][1].setText(test.replace(":", "-"))
                    # Validate IP
                    elif req_config_setting == "UDP_IP":
                        valid = valid_ip(test)
                    # Validate port
                    elif req_config_setting == "UDP_PORT":
                        valid = valid_int(test)
                    else:
                        valid = True
                    req_valid_inputs[req_config_setting] = valid
                    update_req_box_highlight(req_config_setting, (1 if valid else 2) if test else 0)
            # Raspberry Pi
            elif current_device == "RaspberryPi":
                for req_config_setting in config.device_req_config[current_device]:
                    test = req_widgets[current_device][req_config_setting][1].text()
                    # Validate LED Pin
                    if req_config_setting == "LED_PIN":
                        valid = valid_int(test)
                    # Validate LED Freq
                    elif req_config_setting == "LED_FREQ_HZ":
                        valid = valid_int(test)
                    # Validate LED DMA
                    elif req_config_setting == "LED_DMA":
                        valid = valid_int(test)
                    else:
                        valid = True
                    req_valid_inputs[req_config_setting] = valid
                    update_req_box_highlight(req_config_setting, (1 if valid else 2) if test else 0)
            # Fadecandy
            elif current_device == "Fadecandy":
                for req_config_setting in config.device_req_config[current_device]:
                    test = req_widgets[current_device][req_config_setting][1].text()
                    # Validate Server
                    if req_config_setting == "SERVER":
                        valid = validate_address_port(test)
                    else:
                        valid = True
                    req_valid_inputs[req_config_setting] = valid
                    update_req_box_highlight(req_config_setting, (1 if valid else 2) if test else 0)
            elif current_device == "sACNClient":
                for req_config_setting in config.device_req_config[current_device]:
                    test = req_widgets[current_device][req_config_setting][1].text()
                    # Validate Server
                    if req_config_setting == "IP":
                        valid = valid_ip(test)
                    else:
                        valid = True
                    req_valid_inputs[req_config_setting] = valid
                    update_req_box_highlight(req_config_setting, (1 if valid else 2) if test else 0)
            # Other devices without required config
            elif not config.device_req_config[current_device]:
                pass
            for gen_config_setting in config.device_gen_config:
                test = gen_widgets[gen_config_setting][1].text()
                # Validate Server
                if gen_config_setting in ["N_PIXELS", "N_FFT_BINS", "MIN_FREQUENCY", "MAX_FREQUENCY"]:
                    valid = valid_int(test)
                else:
                    valid = True
                gen_valid_inputs[gen_config_setting] = valid
                update_gen_box_highlight(gen_config_setting, (1 if valid else 2) if test else 0)
            update_add_device_button()

        # Set up window and layout
        self.device_dialogue = QDialog(None, Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        self.device_dialogue.setWindowTitle("LED Strip Manager")
        self.device_dialogue.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout()
        self.device_dialogue.setLayout(layout)

        # Set up tab layouts
        tabs = QTabWidget()
        layout.addWidget(tabs)
        addDeviceTab = QWidget()
        remDeviceTab = QWidget()
        addDeviceTabLayout = QVBoxLayout()
        remDeviceTabLayout = QVBoxLayout()
        addDeviceReqGroupBox = QGroupBox("Device Setup")
        addDeviceGenGroupBox = QGroupBox("Configuration")
        remDeviceGroupBox = QGroupBox("Devices")
        addDeviceTabReqButtonLayout = QGridLayout()
        addDeviceTabGenButtonLayout = QGridLayout()
        remDeviceTabButtonLayout = QGridLayout()
        addDeviceReqGroupBox.setLayout(addDeviceTabReqButtonLayout)
        addDeviceGenGroupBox.setLayout(addDeviceTabGenButtonLayout)
        remDeviceGroupBox.setLayout(remDeviceTabButtonLayout)
        addDeviceTab.setLayout(addDeviceTabLayout)
        remDeviceTab.setLayout(remDeviceTabLayout)
        tabs.addTab(addDeviceTab, "Add Device")
        tabs.addTab(remDeviceTab, "Remove Device")

        # Set up "Add Device" tab
        device_type_cbox = QComboBox()
        device_type_cbox.addItems(config.device_req_config.keys())
        device_type_cbox.currentIndexChanged.connect(show_hide_addBoard_interface)
        device_type_cbox.currentIndexChanged.connect(validate_inputs)
        addDeviceTabLayout.addWidget(device_type_cbox)

        # Set up "Add Device" required settings widgets
        req_widgets = {}
        addDeviceTabLayout.addWidget(addDeviceReqGroupBox)
        # if the new board has required config
        for device in config.device_req_config:
            # Make the req_widgets
            req_widgets[device] = {}
            if config.device_req_config[device]:
                for req_config_setting in config.device_req_config[device]:
                    label = config.device_req_config[device][req_config_setting][0]
                    guide = config.device_req_config[device][req_config_setting][1]
                    wType = config.device_req_config[device][req_config_setting][2]
                    deflt = config.device_req_config[device][req_config_setting][3]
                    wLabel = QLabel(label)
                    #wGuide = QLabel(guide)
                    if wType in ["textbox", "textbox-int"]:
                        wEdit = QLineEdit()
                        wEdit.setPlaceholderText(deflt)
                        wEdit.textChanged.connect(validate_inputs)
                    elif wType == "checkbox":
                        wEdit = QCheckBox()
                        wEdit.setCheckState(Qt.Checked if deflt else Qt.Unchecked)
                    req_widgets[device][req_config_setting] = [wLabel, wEdit]
                # Add req_widgets to layout
                i = 0
                for req_config in req_widgets[device]:
                    addDeviceTabReqButtonLayout.addWidget(req_widgets[device][req_config][0], i, 0)
                    addDeviceTabReqButtonLayout.addWidget(req_widgets[device][req_config][1], i, 1)
                    #addDeviceTabReqButtonLayout.addWidget(widget_set[2], i+1, 0, 1, 2)
                    i += 1
            else:
                no_setup = QLabel("Device requires no additional setup here! :)")
                req_widgets[device]["no_config"] = no_setup
                addDeviceTabReqButtonLayout.addWidget(no_setup, 0, 0)

        # Set up "Add Device" general settings widgets
        gen_widgets = {}
        addDeviceTabLayout.addWidget(addDeviceGenGroupBox)
        addDeviceTabLayout.addStretch(1)
        for gen_config_setting in config.device_gen_config:
            label = config.device_gen_config[gen_config_setting][0]
            guide = config.device_gen_config[gen_config_setting][1]
            wType = config.device_gen_config[gen_config_setting][2]
            deflt = config.device_gen_config[gen_config_setting][3]
            wLabel = QLabel(label)
            #wGuide = QLabel(guide)
            if wType in ["textbox", "textbox-int"]:
                wEdit = QLineEdit()
                wEdit.setPlaceholderText(deflt)
                wEdit.textChanged.connect(validate_inputs)
            elif wType == "checkbox":
                wEdit = QCheckBox()
                wEdit.setCheckState(Qt.Checked if deflt else Qt.Unchecked)
            gen_widgets[gen_config_setting] = [wLabel, wEdit]
        # Add gen_widgets to layout
        i = 0
        for req_config in gen_widgets:
            addDeviceTabGenButtonLayout.addWidget(gen_widgets[req_config][0], i, 0)
            addDeviceTabGenButtonLayout.addWidget(gen_widgets[req_config][1], i, 1)
            #addDeviceTabGenButtonLayout.addWidget(widget_set[2], i+1, 0, 1, 2)
            i += 1

        # Show appropriate req_widgets
        show_hide_addBoard_interface()

        self.add_device_button = QPushButton("Add Device")
        self.add_device_button.setEnabled(False)
        self.add_device_button.clicked.connect(addBoard_to_manager)
        addDeviceTabLayout.addWidget(self.add_device_button)

        # Set up "Remove Device" tab
        remDeviceTabLayout.addWidget(remDeviceGroupBox)
        remDeviceTabLayout.addStretch(1)
        # Show devices available to delete
        populate_remove_device_list()

        self.rem_device_button = QPushButton("Delete Device")
        self.rem_device_button.setEnabled(False)
        self.rem_device_button.clicked.connect(remBoard_from_manager)
        remDeviceTabLayout.addWidget(self.rem_device_button)

        # Set up ok/cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.device_dialogue.accept)
        self.buttons.rejected.connect(self.device_dialogue.reject)
        layout.addWidget(self.buttons)
        
        # Set button states and show dialogue
        validate_inputs()
        self.device_dialogue.show()

    def saveDialogue(self):
        # Save window state
        settings.beginGroup("MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue('state', self.saveState())
        settings.endGroup()
        # save all settings
        settings.setValue("settings_dict", config.settings)
        # save and close
        settings.sync()
        # Confirmation message
        self.conf_dialogue = QMessageBox()
        self.conf_dialogue.setText("Settings saved.\nSettings are also automatically saved when program closes.")
        self.conf_dialogue.show()

    def guiDialogue(self):
        def update_visibilty_dict():
            for checkbox in self.gui_vis_checkboxes:
                config.settings["GUI_opts"][checkbox] = self.gui_vis_checkboxes[checkbox].isChecked()
            self.updateUIVisibleItems()

        self.gui_dialogue = QDialog(None, Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        self.gui_dialogue.setWindowTitle("Show/hide Sections")
        self.gui_dialogue.setWindowModality(Qt.ApplicationModal)
        layout = QGridLayout()
        self.gui_dialogue.setLayout(layout)
        # OK button
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.gui_dialogue.accept)

        self.gui_vis_checkboxes = {}
        for section in self.gui_widgets:
            self.gui_vis_checkboxes[section] = QCheckBox(section)
            self.gui_vis_checkboxes[section].setCheckState(
                    Qt.Checked if config.settings["GUI_opts"][section] else Qt.Unchecked)
            self.gui_vis_checkboxes[section].stateChanged.connect(update_visibilty_dict)
            layout.addWidget(self.gui_vis_checkboxes[section])
        layout.addWidget(self.buttons)
        self.gui_dialogue.show()

    def updateColourDropdowns(self):
        colours = colour_manager.getColours("all")
        # dd, dropdown
        for dd in self.colour_dropdowns:
            current = config.settings["devices"][dd[0]]["effect_opts"][dd[1]][dd[2]]
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].clear()
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].addItems(colours)
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].setCurrentIndex(list(colours).index(current))

    def updateGradientDropdowns(self):
        gradients = colour_manager.getGradients()
        for dd in self.gradient_dropdowns:
            current = config.settings["devices"][dd[0]]["effect_opts"][dd[1]][dd[2]]
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].clear()
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].addItems(gradients)
            self.board_tabs_widgets[dd[0]]["grid_layout_widgets"][dd[1]][dd[2]].setCurrentIndex(list(gradients).index(current))

    def initBoardUI(self, board):
        self.board = board
        # Set up wrapping layout
        self.board_tabs_widgets[board]["wrapper"] = QVBoxLayout()
        
        # Set up graph layout
        self.board_tabs_widgets[board]["graph_view"] = pg.GraphicsView()
        graph_layout = pg.GraphicsLayout(border=(100,100,100))
        self.board_tabs_widgets[board]["graph_view"].setCentralItem(graph_layout)
        # Mel filterbank plot
        fft_plot = graph_layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, config.settings["devices"][self.board]["configuration"]["N_FFT_BINS"] + 1))
        self.board_tabs_widgets[board]["mel_curve"] = pg.PlotCurveItem()
        self.board_tabs_widgets[board]["mel_curve"].setData(x=x_data, y=x_data*0)
        fft_plot.addItem(self.board_tabs_widgets[board]["mel_curve"])
        # Visualization plot
        graph_layout.nextRow()
        led_plot = graph_layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        # Pen for each of the color channel curves
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        # Color channel curves
        self.board_tabs_widgets[board]["r_curve"] = pg.PlotCurveItem(pen=r_pen)
        self.board_tabs_widgets[board]["g_curve"] = pg.PlotCurveItem(pen=g_pen)
        self.board_tabs_widgets[board]["b_curve"] = pg.PlotCurveItem(pen=b_pen)
        # Define x data
        x_data = np.array(range(1, config.settings["devices"][self.board]["configuration"]["N_PIXELS"] + 1))
        self.board_tabs_widgets[board]["r_curve"].setData(x=x_data, y=x_data*0)
        self.board_tabs_widgets[board]["g_curve"].setData(x=x_data, y=x_data*0)
        self.board_tabs_widgets[board]["b_curve"].setData(x=x_data, y=x_data*0)
        # Add curves to plot
        led_plot.addItem(self.board_tabs_widgets[board]["r_curve"])
        led_plot.addItem(self.board_tabs_widgets[board]["g_curve"])
        led_plot.addItem(self.board_tabs_widgets[board]["b_curve"])

        # Set up button layout
        self.board_tabs_widgets[board]["label_reactive"] = QLabel("Audio Reactive Effects")
        self.board_tabs_widgets[board]["label_non_reactive"] = QLabel("Non Reactive Effects")
        self.board_tabs_widgets[board]["reactive_button_grid_wrap"] = QWidget()
        self.board_tabs_widgets[board]["non_reactive_button_grid_wrap"] = QWidget()
        self.board_tabs_widgets[board]["reactive_button_grid"] = QGridLayout()
        self.board_tabs_widgets[board]["non_reactive_button_grid"] = QGridLayout()
        self.board_tabs_widgets[board]["reactive_button_grid_wrap"].setLayout(self.board_tabs_widgets[board]["reactive_button_grid"])   
        self.board_tabs_widgets[board]["non_reactive_button_grid_wrap"].setLayout(self.board_tabs_widgets[board]["non_reactive_button_grid"])   
        buttons = {}
        connecting_funcs = {}
        grid_width = 4
        i = 0
        j = 0
        k = 0
        l = 0
        # Dynamically layout reactive_buttons and connect them to the visualisation effects
        def connect_generator(effect):
            def func():
                config.settings["devices"][board]["configuration"]["current_effect"] = effect
                buttons[effect].setDown(True)
            func.__name__ = effect
            return func
        # Where the magic happens
        for effect in board_manager.visualizers[board].effects:
            if not effect in board_manager.visualizers[board].non_reactive_effects:
                connecting_funcs[effect] = connect_generator(effect)
                buttons[effect] = QPushButton(effect)
                buttons[effect].clicked.connect(connecting_funcs[effect])
                self.board_tabs_widgets[board]["reactive_button_grid"].addWidget(buttons[effect], j, i)
                i += 1
                if i % grid_width == 0:
                    i = 0
                    j += 1
            else:
                connecting_funcs[effect] = connect_generator(effect)
                buttons[effect] = QPushButton(effect)
                buttons[effect].clicked.connect(connecting_funcs[effect])
                self.board_tabs_widgets[board]["non_reactive_button_grid"].addWidget(buttons[effect], l, k)
                k += 1
                if k % grid_width == 0:
                    k = 0
                    l += 1
                
        # Set up frequency slider
        # Frequency range label
        self.board_tabs_widgets[board]["label_slider"] = QLabel("Frequency Range")
        # Frequency slider
        def freq_slider_change(tick):
            minf = self.board_tabs_widgets[board]["freq_slider"].tickValue(0)**2.0 * (config.settings["mic_config"]["MIC_RATE"] / 2.0)
            maxf = self.board_tabs_widgets[board]["freq_slider"].tickValue(1)**2.0 * (config.settings["mic_config"]["MIC_RATE"] / 2.0)
            t = 'Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf)
            freq_label.setText(t)
            config.settings["devices"][self.board]["configuration"]["MIN_FREQUENCY"] = minf
            config.settings["devices"][self.board]["configuration"]["MAX_FREQUENCY"] = maxf
            board_manager.signal_processers[self.board].create_mel_bank()
        def set_freq_min():
            config.settings["devices"][board]["configuration"]["MIN_FREQUENCY"] = self.board_tabs_widgets[board]["freq_slider"].start()
            board_manager.signal_processers[board].create_mel_bank()
        def set_freq_max():
            config.settings["devices"][board]["configuration"]["MAX_FREQUENCY"] = self.board_tabs_widgets[board]["freq_slider"].end()
            board_manager.signal_processers[board].create_mel_bank()
        self.board_tabs_widgets[board]["freq_slider"] = QRangeSlider()
        self.board_tabs_widgets[board]["freq_slider"].show()
        self.board_tabs_widgets[board]["freq_slider"].setMin(0)
        self.board_tabs_widgets[board]["freq_slider"].setMax(20000)
        self.board_tabs_widgets[board]["freq_slider"].setRange(config.settings["devices"][board]["configuration"]["MIN_FREQUENCY"], config.settings["devices"][board]["configuration"]["MAX_FREQUENCY"])
        self.board_tabs_widgets[board]["freq_slider"].setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        self.board_tabs_widgets[board]["freq_slider"].setSpanStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
        self.board_tabs_widgets[board]["freq_slider"].setDrawValues(True)
        self.board_tabs_widgets[board]["freq_slider"].endValueChanged.connect(set_freq_max)
        self.board_tabs_widgets[board]["freq_slider"].startValueChanged.connect(set_freq_min)
        self.board_tabs_widgets[board]["freq_slider"].setStyleSheet("""
        QRangeSlider * {
            border: 0px;
            padding: 0px;
        }
        QRangeSlider > QSplitter::handle {
            background: #fff;
        }
        QRangeSlider > QSplitter::handle:vertical {
            height: 3px;
        }
        QRangeSlider > QSplitter::handle:pressed {
            background: #ca5;
        }
        """)

        # Set up option tabs layout
        self.board_tabs_widgets[board]["label_options"] = QLabel("Effect Options")
        self.board_tabs_widgets[board]["opts_tabs"] = QTabWidget()
        # Dynamically set up tabs
        tabs = {}
        grid_layouts = {}
        self.board_tabs_widgets[board]["grid_layout_widgets"] = {}
        # easy access to dropdowns of colours or gradients. they often need updating.
        # contains tuples (board, effect, key) to allow access in self.board_tabs_widgets
        self.colour_dropdowns = []
        self.gradient_dropdowns = []
        options = config.settings["devices"][board]["effect_opts"].keys()
        for effect in board_manager.visualizers[self.board].effects:
            # Make the tab
            self.board_tabs_widgets[board]["grid_layout_widgets"][effect] = {}
            tabs[effect] = QWidget()
            grid_layouts[effect] = QGridLayout()
            tabs[effect].setLayout(grid_layouts[effect])
            self.board_tabs_widgets[board]["opts_tabs"].addTab(tabs[effect],effect)
            # These functions make functions for the dynamic ui generation
            # YOU WANT-A DYNAMIC I GIVE-A YOU DYNAMIC!
            def gen_slider_valuechanger(effect, key):
                def func():
                    config.settings["devices"][board]["effect_opts"][effect][key] = self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].value()
                return func
            def gen_float_slider_valuechanger(effect, key):
                def func():
                    config.settings["devices"][board]["effect_opts"][effect][key] = self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].slider_value
                return func
            def gen_combobox_valuechanger(effect, key):
                def func():
                    config.settings["devices"][board]["effect_opts"][effect][key] = self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].currentText()
                return func
            def gen_checkbox_valuechanger(effect, key):
                def func():
                    config.settings["devices"][board]["effect_opts"][effect][key] = self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].isChecked()
                return func
            # Dynamically generate ui for settings
            if effect in config.dynamic_effects_config:
                i = 0
                connecting_funcs[effect] = {}
                for key, label, ui_element, *opts in config.dynamic_effects_config[effect]:
                    if opts: # neatest way  ^^^^^ i could think of to unpack and handle an unknown number of opts (if any) NOTE only works with py >=3.6
                        if opts[0] not in ["colours", "gradients"]:
                            opts = list(opts[0])
                    if ui_element == "slider":
                        connecting_funcs[effect][key] = gen_slider_valuechanger(effect, key)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key] = QSlider(Qt.Horizontal)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setMinimum(opts[0])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setMaximum(opts[1])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setValue(config.settings["devices"][board]["effect_opts"][effect][key])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].valueChanged.connect(connecting_funcs[effect][key])
                    elif ui_element == "float_slider":
                        connecting_funcs[effect][key] = gen_float_slider_valuechanger(effect, key)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key] = QFloatSlider(*opts, config.settings["devices"][board]["effect_opts"][effect][key])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setValue(config.settings["devices"][board]["effect_opts"][effect][key])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].valueChanged.connect(connecting_funcs[effect][key])
                    elif ui_element == "dropdown":
                        if opts[0] == "colours":
                            self.colour_dropdowns.append((board, effect, key))
                            opts = list(colour_manager.getColours("all"))
                        elif opts[0] == "gradients":
                            self.gradient_dropdowns.append((board, effect, key))
                            opts = list(colour_manager.getGradients("all"))
                        connecting_funcs[effect][key] = gen_combobox_valuechanger(effect, key)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key] = QComboBox()
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].addItems(opts)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setCurrentIndex(opts.index(config.settings["devices"][board]["effect_opts"][effect][key]))
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].currentTextChanged.connect(connecting_funcs[effect][key])
                    elif ui_element == "checkbox":
                        connecting_funcs[effect][key] = gen_checkbox_valuechanger(effect, key)
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key] = QCheckBox()
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].stateChanged.connect(
                                connecting_funcs[effect][key])
                        self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key].setCheckState(
                                Qt.Checked if config.settings["devices"][board]["effect_opts"][effect][key] else Qt.Unchecked)
                    grid_layouts[effect].addWidget(QLabel(label),i,0)
                    grid_layouts[effect].addWidget(self.board_tabs_widgets[board]["grid_layout_widgets"][effect][key],i,1)
                    i += 1    
            else:
                grid_layouts[effect].addWidget(QLabel("No customisable options for this effect :("),0,0)
                
        
        
        # Add layouts into self.board_tabs_widgets[board]["wrapper"]
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["graph_view"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["label_reactive"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["reactive_button_grid_wrap"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["label_non_reactive"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["non_reactive_button_grid_wrap"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["label_slider"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["freq_slider"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["label_options"])
        self.board_tabs_widgets[board]["wrapper"].addWidget(self.board_tabs_widgets[board]["opts_tabs"])


def update_config_dicts():
    # Updates config.settings with any values stored in settings.ini
    if settings.value("settings_dict"):
        for settings_dict in settings.value("settings_dict"):
            if not config.use_defaults[settings_dict]:
                try:
                    config.settings[settings_dict] = {**config.settings[settings_dict], **settings.value("settings_dict")[settings_dict]}
                except TypeError:
                    print("Error parsing settings dictionary {}".format(settings_dict))
                    pass
    else:
        print("Could not find settings.ini")

def save_config_dicts():
    # saves config.settings
    settings.setValue("settings_dict", config.settings)
    settings.sync()

def frames_per_second():
    """ Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)

def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)

def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    
    audio_datas = {}        
    outputs = {}
    audio_input = True

    # Visualization for each board
    for board in board_manager.boards:
        # Get processed audio data for each device
        audio_datas[board] = board_manager.signal_processers[board].update(audio_samples)
        # Get visualization output for each board
        audio_input = audio_datas[board]["vol"] > config.settings["configuration"]["MIN_VOLUME_THRESHOLD"]
        outputs[board] = board_manager.visualizers[board].get_vis(audio_datas[board]["mel"], audio_input)
        # Map filterbank output onto LED strip(s)
        board_manager.boards[board].show(outputs[board])
        if config.settings["configuration"]["USE_GUI"]:
            # Plot filterbank output
            gui.board_tabs_widgets[board]["mel_curve"].setData(x=audio_datas[board]["x"], y=audio_datas[board]["y"])
            # Plot visualizer output
            gui.board_tabs_widgets[board]["r_curve"].setData(y=outputs[board][0])
            gui.board_tabs_widgets[board]["g_curve"].setData(y=outputs[board][1])
            gui.board_tabs_widgets[board]["b_curve"].setData(y=outputs[board][2])

    # FPS update
    fps = frames_per_second()
    if time.time() - 0.5 > prev_fps_update:
        prev_fps_update = time.time()

    # Various GUI updates
    if config.settings["configuration"]["USE_GUI"]:
        # Update error label
        if audio_input:
            gui.label_error.setText("")
        else:
            gui.label_error.setText("No audio input. Volume below threshold.")
        app.processEvents()

    # Left in just in case prople dont use the gui
    elif audio_datas[board]["vol"] < config.settings["configuration"]["MIN_VOLUME_THRESHOLD"]:
        print("No audio input. Volume below threshold. Volume: {}".format(audio_datas[board]["vol"]))
    if config.settings["configuration"]["DISPLAY_FPS"]:
        print('FPS {:.0f} / {:.0f}'.format(fps, config.settings["configuration"]["FPS"]))

# Load and update configuration from settings.ini
settings = QSettings('./lib/settings.ini', QSettings.IniFormat)
settings.setFallbacksEnabled(False)    # File only, no fallback to registry
update_config_dicts()

# Initialise board(s)
board_manager = BoardManager()


# Initialise Colour Manager
colour_manager = ColourManager()

# Initialise GUI 
if config.settings["configuration"]["USE_GUI"]:
    # Create GUI window
    app = QApplication([])
    app.setApplicationName('Visualization')
    gui = GUI()
    app.processEvents()

# Populate board manager with boards
for board in config.settings["devices"]:
    board_manager.addBoard(board, config_exists=True)

# FPS 
prev_fps_update = time.time()
# The previous time that the frames_per_second() function was called
_time_prev = time.time() * 1000.0
# The low-pass filter used to estimate frames-per-second
_fps = dsp.ExpFilter(val=config.settings["configuration"]["FPS"], alpha_decay=0.2, alpha_rise=0.2)

# Start listening to live audio stream
py_audio = pyaudio.PyAudio()
microphone = Microphone(microphone_update)
try:
    microphone.startStream()
finally:
    save_config_dicts()
    colour_manager.saveColours()
    colour_manager.saveGradients()
    py_audio.terminate()