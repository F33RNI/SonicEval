"""
 Copyright (C) 2022 Fern Lane, Pulsely project
 Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
       https://www.gnu.org/licenses/agpl-3.0.en.html
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
"""
import array
import math
import threading
import traceback

import numpy as np
import pyaudio
from PyQt5 import QtCore
from scipy.signal import butter, lfilter

MEASURE_LATENCY_MAX_LATENCY_CHUNKS = 64
MEASURE_LATENCY_VOLUME = 0.8
MEASURE_LATENCY_FREQ_1 = 1000.
MEASURE_LATENCY_FREQ_2 = 440.

# Defines
DEVICE_TYPE_INPUT = 0
DEVICE_TYPE_OUTPUT = 1
TEST_SIGNAL_TYPE_SWEEP = 0
TEST_SIGNAL_TYPE_NOISE = 1
WINDOW_TYPE_NONE = 0
WINDOW_TYPE_HAMMING = 1
WINDOW_TYPE_HANNING = 2
WINDOW_TYPE_BLACKMAN = 3
MEASURE_LATENCY_STAGE_FREQ_1 = 0
MEASURE_LATENCY_STAGE_FREQ_2 = 1
POSITION_EQUAL = 0
POSITION_ON_LEFT = 1
POSITION_ON_RIGHT = 2


def compute_fft_dbfs(data, window: np.ndarray, window_type: int, signal_type=TEST_SIGNAL_TYPE_SWEEP):
    """
    Computes real fft in dBFS
    TODO: Make proper window compensation
    :param data: input data (float32)
    :param window: fft window (None, np.hamming, np.hanning or np.blackman)
    :param window_type: WINDOW_TYPE_NONE or WINDOW_TYPE_HAMMING or WINDOW_TYPE_HANNING or WINDOW_TYPE_BLACKMAN
    :param signal_type: TEST_SIGNAL_TYPE_SWEEP or TEST_SIGNAL_TYPE_NOISE to apply amplitude correction
    :return: fft in dBFS
    """

    # Multiply by a window
    if window is not None:
        data = data[0:len(data)] * window

    # Calculate real FFT
    real_fft = np.fft.rfft(data)

    # Scale the magnitude of FFT by window and factor of 2
    if window is not None:
        s_mag = np.abs(real_fft) * 2 / np.sum(window)
    else:
        s_mag = np.abs(real_fft) * 2 / (len(data) / 2)

    # Get window compensation factor
    if window_type == WINDOW_TYPE_HAMMING:
        window_power_bandwidth = 1.36
    elif window_type == WINDOW_TYPE_HANNING:
        window_power_bandwidth = 1.50
    elif window_type == WINDOW_TYPE_BLACKMAN:
        window_power_bandwidth = 1.73
    else:
        window_power_bandwidth = 1.

    # Correct signal amplitude
    if signal_type == TEST_SIGNAL_TYPE_SWEEP:
        s_mag *= math.sqrt(window_power_bandwidth)
    else:
        s_mag *= math.sqrt(len(data) * window_power_bandwidth)

    # Convert to dBFS
    return 20 * np.log10(s_mag)


def generate_window(window_type: int, length: int):
    """
    Generates window for FFT
    :param window_type: WINDOW_TYPE_NONE or WINDOW_TYPE_HAMMING or WINDOW_TYPE_HANNING or WINDOW_TYPE_BLACKMAN
    :param length: size of data
    :return: numpy window
    """
    # Create window
    if window_type == WINDOW_TYPE_HAMMING:
        window = np.hamming(length)
    elif window_type == WINDOW_TYPE_HANNING:
        window = np.hanning(length)
    elif window_type == WINDOW_TYPE_BLACKMAN:
        window = np.blackman(length)
    else:
        window = None
    return window


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Bandpass filter
    :param data: data chunk
    :param lowcut: low frequency
    :param highcut: upper frequency
    :param fs: sample rate
    :param order: order of filter
    :return: signal after bandpass
    """
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = lfilter(b, a, data)
    return y


def apply_reference(input_frequencies, input_levels, reference_frequencies, reference_levels, normalize_ref=False):
    """
    Applies and interpolates reference to signal
    TODO: Optimise this shit
    :param input_frequencies: [f1, f2, f3]
    :param input_levels: [[ch1_f1_lvl, ch1_f2_lvl], [ch2_f1_lvl, ch2_f2_lvl]]
    :param reference_frequencies: [f1, f2, f3]
    :param reference_levels: [[ch1_f1_lvl, ch1_f2_lvl], [ch2_f1_lvl, ch2_f2_lvl]]
    :return: [[ch1_f1_lvl, ch1_f2_lvl], [ch2_f1_lvl, ch2_f2_lvl]]
    :param normalize_ref: normalize reference data before applying it
    """
    # Interpolated reference
    output_levels = np.array(input_levels).copy()

    # Find how far is reference_frequencies from input_frequencies start
    from_start_to_ref_indexes = calculate_distance_indexes(input_frequencies, reference_frequencies)

    # Find how far is reference_frequencies from input_frequencies end
    from_end_to_ref_indexes = calculate_distance_indexes(input_frequencies, reference_frequencies, False)

    # Find how far is input_frequencies from reference_frequencies start
    from_ref_to_start_indexes = calculate_distance_indexes(reference_frequencies, input_frequencies)

    # Find how far is input_frequencies from reference_frequencies end
    from_ref_to_end_indexes = calculate_distance_indexes(reference_frequencies, input_frequencies, False)

    # Fill start gap with first reference value
    if from_start_to_ref_indexes is not None and from_start_to_ref_indexes > 0:
        reference_levels_interpolated_start = np.ones((len(input_levels), from_start_to_ref_indexes), dtype=float) \
                                              * reference_levels.transpose()[0][:, None]
    else:
        reference_levels_interpolated_start = np.empty((len(input_levels), 0), dtype=float)

    # Fill end gap with last reference value
    if from_end_to_ref_indexes is not None and from_end_to_ref_indexes < 0:
        reference_levels_interpolated_end = np.ones((len(input_levels), abs(from_end_to_ref_indexes)), dtype=float) \
                                            * reference_levels.transpose()[-1][:, None]
    else:
        reference_levels_interpolated_end = np.empty((len(input_levels), 0), dtype=float)

    # Calculate middle part length of interpolated reference signal
    middle_length_target = len(input_frequencies)
    if from_start_to_ref_indexes is not None and from_start_to_ref_indexes > 0:
        middle_length_target -= from_start_to_ref_indexes
    if from_end_to_ref_indexes is not None and from_end_to_ref_indexes < 0:
        middle_length_target -= abs(from_end_to_ref_indexes)

    reference_cut_index_start = 0
    if from_ref_to_start_indexes is not None and from_ref_to_start_indexes > 0:
        reference_cut_index_start += from_ref_to_start_indexes

    reference_cut_index_end = len(reference_frequencies)
    if from_ref_to_end_indexes is not None and from_ref_to_end_indexes < 0:
        reference_cut_index_end -= abs(from_ref_to_end_indexes)

    # Interpolate middle part
    reference_levels_interpolated_middle = []
    for channel_n in range(len(reference_levels)):
        # Stretch each channel to middle_length_target
        reference_levels_interpolated_middle.append(stretch_to(list(
            reference_levels[channel_n][reference_cut_index_start: reference_cut_index_end]), middle_length_target))
    reference_levels_interpolated_middle = np.array(reference_levels_interpolated_middle)

    # Build final interpolated reference
    if len(reference_levels_interpolated_start[0]) > 0:
        reference_levels_interpolated = np.append(reference_levels_interpolated_start,
                                                  reference_levels_interpolated_middle, axis=1)
    else:
        reference_levels_interpolated = reference_levels_interpolated_middle
    if len(reference_levels_interpolated_end[0]) > 0:
        reference_levels_interpolated = np.append(reference_levels_interpolated,
                                                  reference_levels_interpolated_end, axis=1)

    # Normalize?
    if normalize_ref:
        reference_levels_interpolated = normalize_data(reference_levels_interpolated)

    # output = input - reference
    output_levels = np.subtract(output_levels, reference_levels_interpolated)

    return output_levels


def calculate_distance_indexes(input_frequencies, reference_frequencies, calculate_start=True):
    """
    Find how far is reference signal far from input_frequencies start / end
    :param input_frequencies:
    :param reference_frequencies:
    :param calculate_start:
    :return:
    """
    # Find reference position
    if reference_frequencies[0 if calculate_start else - 1] > input_frequencies[0 if calculate_start else - 1]:
        reference_position = POSITION_ON_RIGHT
    elif reference_frequencies[0 if calculate_start else - 1] < input_frequencies[0 if calculate_start else - 1]:
        reference_position = POSITION_ON_LEFT
    else:
        reference_position = POSITION_EQUAL

    # Find how far is reference signal far from input_frequencies start
    from_input_to_ref_indexes = 0
    if reference_position != POSITION_EQUAL:
        if reference_position == POSITION_ON_RIGHT:
            # signal___===... (signal starts before ref)
            if calculate_start:
                for frequency in input_frequencies:
                    if reference_frequencies[0] > frequency:
                        from_input_to_ref_indexes += 1

            # ...===ref--- (signal ends before ref)
            else:
                from_input_to_ref_indexes = None

        elif reference_position == POSITION_ON_LEFT:
            # ref---===... (ref starts before signal)
            if calculate_start:
                from_input_to_ref_indexes = None

            # ...===signal___ (ref ends before signal)
            else:
                for frequency in np.flip(input_frequencies):
                    if reference_frequencies[-1] < frequency:
                        from_input_to_ref_indexes -= 1

    return from_input_to_ref_indexes


def normalize_data(data):
    """
    Moves data to 0 dBFS
    :param data: data numpy array to normalize
    :return: normalized data numpy array
    """
    data_norm = data.copy()
    return np.add(data_norm, -np.max(data_norm))


def frequency_to_index(frequency, sample_rate: int, data_length: int):
    """
    Converts frequency in Hz to index in FFT array
    :param frequency: frequency in Hz
    :param sample_rate: sample rate
    :param data_length: input fft data length (fft output length * 2)
    :return:
    """
    return int(_map(frequency, 0, sample_rate / 2, 0, data_length / 2))


def index_to_frequency(index, sample_rate: int, data_length: int):
    """
    Converts index in FFT array to frequency in Hz
    :param index: index in final fft array
    :param sample_rate: sample rate
    :param data_length: input fft data length (fft output length * 2)
    :return: frequency in Hz
    """
    return int(_map(index, 0, data_length / 2, 0, sample_rate / 2))


def _map(x, in_min, in_max, out_min, out_max):
    """
    Arduino map function
    :param x:
    :param in_min:
    :param in_max:
    :param out_min:
    :param out_max:
    :return:
    """
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def find_nearest(array_, value):
    """
    Finds closes value in given array
    :param array_: numpy array
    :param value: value
    :return:
    """
    array_ = np.asarray(array_)
    idx = (np.abs(array_ - value)).argmin()
    return array_[idx]


def sample_rate_to_chunk_size(sample_rate):
    """
    Calculates chunk size for given sample rate
    :param sample_rate:
    :return:
    """
    chunk_sizes = 2 ** np.arange(8, 14)
    return find_nearest(chunk_sizes, sample_rate // 32)


def stretch_to(list_, target_length: int):
    """
    Stretches list to length
    :param list_: source list
    :param target_length: target length
    :return: stretched list
    """
    # Create new list with target length
    out = [None] * target_length

    # Measure source length
    input_length = len(list_)

    # Map target list
    if input_length > 1:
        for i, x in enumerate(list_):
            out[i * (target_length - 1) // (input_length - 1)] = x

    value = list_[0]

    # Fill Nones with prev. values
    for i in range(len(out)):
        if out[i] is None:
            out[i] = value
        else:
            value = out[i]

    return out


def clamp(n, min_, max_):
    """
    Clamps number to range
    :param n: number to clamp
    :param min_: minimum allowed value
    :param max_: maximum allowed value
    :return: clamped value
    """
    return max(min(max_, n), min_)


class AudioHandler:
    def __init__(self, settings_handler):
        """
        Initializes AudioHandler class
        :param settings_handler:
        """
        self.settings_handler = settings_handler

        # Streams (for close function)
        self.playback_stream = None
        self.recording_stream = None

        # Final data
        self.frequency_response_frequencies = []
        self.frequency_response_levels_per_channels = []

        # Reference data
        self.reference_frequencies = []
        self.reference_levels_per_channels = []

        # Class variables
        self.error_message = ''
        self.py_audio = None
        self.measure_latency_thread_running = False
        self.audio_latency_chunks = -1
        self.label_latency_update_signal = None
        self.update_label_info = None
        self.measurement_timer_start_signal = None
        self.chunk_size = 0

    def open_audio(self, recording_channels: int):
        """
        Opens playback and recording streams
        :param recording_channels: number of channels to record
        :return: playback_stream, recording_stream
        """
        playback_device_name = str(self.settings_handler.settings['audio_playback_interface'])
        recording_device_name = str(self.settings_handler.settings['audio_recording_interface'])
        playback_device_index = self.get_device_index_by_name(playback_device_name)
        recording_device_index = self.get_device_index_by_name(recording_device_name)

        # Get sample rate
        sample_rate = int(self.settings_handler.settings['audio_sample_rate'])

        # Calculate chunk size
        self.chunk_size = sample_rate_to_chunk_size(sample_rate)

        # Open playback stream
        self.playback_stream = self.py_audio.open(output_device_index=playback_device_index,
                                                  format=pyaudio.paFloat32,
                                                  channels=1,
                                                  frames_per_buffer=self.chunk_size,
                                                  rate=sample_rate,
                                                  output=True)

        # Open recording stream
        self.recording_stream = self.py_audio.open(input_device_index=recording_device_index,
                                                   format=pyaudio.paFloat32,
                                                   channels=recording_channels,
                                                   frames_per_buffer=self.chunk_size,
                                                   rate=sample_rate,
                                                   input=True)

        # Return streams
        return self.playback_stream, self.recording_stream

    def close_audio(self):
        """
        Closes playback and recording streams
        :return:
        """
        if self.playback_stream is not None:
            self.playback_stream.close()
        if self.recording_stream is not None:
            self.recording_stream.close()

    def initialize_py_audio(self):
        """
        Initializes PyAudio() class object
        :return:
        """
        if self.py_audio is None:
            self.py_audio = pyaudio.PyAudio()
        else:
            self.py_audio.terminate()
            self.py_audio = pyaudio.PyAudio()

    def get_devices_names(self, device_type: int):
        """
        Gets input or output devices names list
        :param device_type: DEVICE_TYPE_INPUT or DEVICE_TYPE_OUTPUT
        :return: list of names
        """
        devices_names = []
        info = self.py_audio.get_host_api_info_by_index(0)

        for i in range(0, info.get('deviceCount')):
            if (self.py_audio.get_device_info_by_host_api_device_index(0, i)
                    .get('maxInputChannels' if device_type == DEVICE_TYPE_INPUT else 'maxOutputChannels')) > 0:
                devices_names.append(self.py_audio.get_device_info_by_host_api_device_index(0, i).get('name'))

        return devices_names

    def get_device_index_by_name(self, device_name: str):
        """
        Gets device index by it's name
        :param device_name: name of device from get_devices_names()
        :return:
        """
        try:
            if self.py_audio is None:
                self.py_audio = pyaudio.PyAudio()
            info = self.py_audio.get_host_api_info_by_index(0)
            device_count = info.get('deviceCount')
            for i in range(0, device_count):
                device = self.py_audio.get_device_info_by_host_api_device_index(0, i)
                if device_name.lower() in str(device.get('name')).lower():
                    return device.get('index')

        except Exception as e:
            print(e)

        return 0

    def measure_latency(self, label_latency_update_signal: QtCore.pyqtSignal,
                        update_label_info: QtCore.pyqtSignal,
                        measurement_timer_start_signal: QtCore.pyqtSignal):
        """
        Starts measure_latency_loop
        :param label_latency_update_signal:
        :param update_label_info:
        :param measurement_timer_start_signal:
        :return:
        """
        self.label_latency_update_signal = label_latency_update_signal
        self.update_label_info = update_label_info
        self.measurement_timer_start_signal = measurement_timer_start_signal

        # Reset error message
        self.error_message = ''

        # Start measuring latency loop
        self.measure_latency_thread_running = True
        threading.Thread(target=self.measure_latency_loop).start()

    def measure_latency_loop(self):
        """
        Measures latency in chunks between playback and receive
        :return:
        """
        try:
            # Loop variables
            phase = 0
            chunks_counter = 0
            chunks_counter_timeout = 0
            measure_latency_stage = MEASURE_LATENCY_STAGE_FREQ_1
            measure_latency_started = False

            # Open audio streams
            playback_stream, recording_stream = self.open_audio(1)

            # Reset latency
            self.audio_latency_chunks = -1

            # Sample rate
            sample_rate = int(self.settings_handler.settings['audio_sample_rate'])

            # FFT window
            window_type = int(self.settings_handler.settings['fft_window_type'])
            window = generate_window(window_type, self.chunk_size)

            # Data buffer (floats)
            samples = np.zeros(self.chunk_size, dtype=np.float)

            while self.measure_latency_thread_running:
                # Current frequency
                if measure_latency_stage == MEASURE_LATENCY_STAGE_FREQ_1:
                    frequency = MEASURE_LATENCY_FREQ_1
                elif measure_latency_stage == MEASURE_LATENCY_STAGE_FREQ_2:
                    frequency = MEASURE_LATENCY_FREQ_2
                else:
                    break

                # Fill buffer with sine wave
                for sample in range(self.chunk_size):
                    samples[sample] = MEASURE_LATENCY_VOLUME * np.sin(phase)
                    phase += 2 * np.pi * frequency / sample_rate

                # Convert to bytes
                output_bytes = array.array('f', samples).tobytes()

                # Write to stream
                playback_stream.write(output_bytes)

                # Read data (1 channel)
                input_data_raw = recording_stream.read(self.chunk_size)
                input_data = np.frombuffer(input_data_raw, dtype=np.float32)

                # Compute FFT
                fft_dbfs = compute_fft_dbfs(input_data, window, window_type)

                # Mean of signal (dbfs)
                fft_mean = np.mean(fft_dbfs)

                # Real peak value and index
                fft_max = np.max(fft_dbfs)
                fft_max_index = np.where(fft_dbfs == fft_max)[0][0]

                # Expected index of peak
                fft_max_expected_index = frequency_to_index(frequency, sample_rate, len(input_data))

                # Expected frequency is detected
                if abs(fft_mean) / abs(fft_max) > 2 and abs(fft_max_index - fft_max_expected_index) < 5.:
                    # Detected 1st frequency -> start counting
                    if measure_latency_stage == MEASURE_LATENCY_STAGE_FREQ_1:
                        measure_latency_started = True

                    # Detected 2nd frequency -> stop counting and exit
                    elif measure_latency_stage == MEASURE_LATENCY_STAGE_FREQ_2:
                        # Stop counter
                        measure_latency_started = False

                        # Store measured latency
                        self.audio_latency_chunks = chunks_counter

                        # Exit
                        self.measure_latency_thread_running = False

                    # Increment stage
                    measure_latency_stage += 1

                # Count latency in chunks
                if measure_latency_started:
                    chunks_counter += 1

                # Show info message
                if self.update_label_info is not None:
                    self.update_label_info.emit('Playing frequency: ' + str(frequency)
                                                + ' Hz, Reading: ' + str(int(fft_dbfs[fft_max_expected_index]))
                                                + ' dBFS, Latency: ' + str(chunks_counter * self.chunk_size)
                                                + ' samples')

                # Timeout error measuring latency
                chunks_counter_timeout += 1
                if chunks_counter_timeout >= MEASURE_LATENCY_MAX_LATENCY_CHUNKS:
                    self.audio_latency_chunks = -1
                    self.measure_latency_thread_running = False
                    break

            # Close audio streams
            self.close_audio()

            # Display measured latency
            if self.label_latency_update_signal is not None:
                if self.audio_latency_chunks >= 0:
                    self.label_latency_update_signal.emit('Latency: ' +
                                                          str(self.audio_latency_chunks * self.chunk_size) +
                                                          ' samples (' + str(round(self.audio_latency_chunks
                                                                                   * self.chunk_size
                                                                                   / sample_rate * 1000, 2)) + ' ms)')
                else:
                    self.label_latency_update_signal.emit('Failed to measure latency!')

                # Exit
                if self.measurement_timer_start_signal is not None:
                    self.measurement_timer_start_signal.emit(1)

        # Error during latency measurement
        except Exception as e:
            traceback.print_exc()
            self.error_message = str(e)
            if self.measurement_timer_start_signal is not None:
                self.measurement_timer_start_signal.emit(1)

    def stop_measuring_latency(self):
        """
        Stops measuring latency
        :return:
        """
        if self.measure_latency_thread_running:
            # Stop loop
            self.measure_latency_thread_running = False

            # Reset latency
            self.audio_latency_chunks = -1

            # Display message
            if self.label_latency_update_signal is not None:
                self.label_latency_update_signal.emit('Latency measurement stopped!')
