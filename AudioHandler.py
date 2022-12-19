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
from ftfy import fix_encoding
from scipy.signal import butter, lfilter, find_peaks

# How long play silence (in chunks) before playing tone
SILENCE_BEFORE_MEASUREMENT_CHUNKS = 64

# Maximum latency (timeout threshold)
MEASURE_LATENCY_MAX_LATENCY_CHUNKS = 64

# Volume to play while detecting latency
MEASURE_LATENCY_VOLUME = 0.8

# Accepted frequency deviation while detecting signal for the first time
MEASURE_LATENCY_ACCEPTED_DEVIATION_HZ = 100.

# Accepted volume range while detecting latency
MEASURE_LATENCY_MIN_VOLUME = .07
MEASURE_LATENCY_MAX_VOLUME = .9

# What can be the maximum allowable difference between the two measured latencies
MEASURE_LATENCY_MAX_TOLERANCE_SAMPLES = 5

# Starting from what threshold (maximum level / threshold) to start detecting peaks
MEASURE_LATENCY_PEAKS_THRESHOLD_VOLUME = 1.32

# Defines
DEVICE_TYPE_INPUT = 0
DEVICE_TYPE_OUTPUT = 1
TEST_SIGNAL_TYPE_SWEEP = 0
TEST_SIGNAL_TYPE_NOISE = 1
WINDOW_TYPE_NONE = 0
WINDOW_TYPE_HAMMING = 1
WINDOW_TYPE_HANNING = 2
WINDOW_TYPE_BLACKMAN = 3
MEASURE_LATENCY_STAGE_1 = 0
MEASURE_LATENCY_STAGE_2 = 1
MEASURE_LATENCY_STAGE_3 = 2
MEASURE_LATENCY_STAGE_4 = 3
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


def is_all_frequencies_accepted(frequency_list, expected_frequency):
    """
    Checks if all elements in given list is deviated from expected_frequency no more than
    MEASURE_LATENCY_ACCEPTED_DEVIATION_HZ
    :param frequency_list:
    :param expected_frequency:
    :return:
    """
    list_accepted = True
    for frequency in frequency_list:
        if abs(frequency - expected_frequency) > MEASURE_LATENCY_ACCEPTED_DEVIATION_HZ:
            list_accepted = False
            break
    return list_accepted


def find_phase_changes_by_peaks(samples, peaks_indexes, test_frequency_samples, positive=True):
    """
    Finds index in samples array where phase changes
    :param samples: audio samples
    :param peaks_indexes: indexes of peaks
    :param test_frequency_samples: test frequency period
    :param positive: positive or negative peaks (for calculating volume)
    :return:
    """
    # Average distance between peaks in samples
    peaks_diff_avg = np.average(np.diff(peaks_indexes))
    peak_sample_n_last = -math.inf

    phase_changes_sample_n = []
    phase_changes_volume = []

    for sample_n in peaks_indexes:
        distance = sample_n - peak_sample_n_last
        deviation = distance - peaks_diff_avg
        if test_frequency_samples / 4 < deviation < test_frequency_samples * 2:
            gap_start_index = sample_n - distance
            gap_stop_index = sample_n

            volume_gap_avg = 0
            for check_index in range(int(gap_start_index), int(gap_stop_index)):
                sample = samples[check_index]
                if positive:
                    if sample > 0.:
                        volume_gap_avg += sample

                else:
                    if sample < 0.:
                        volume_gap_avg += abs(sample)

            volume_gap_avg /= distance

            phase_changes_sample_n.append(int(sample_n - distance / 2))
            phase_changes_volume.append(volume_gap_avg)

        peak_sample_n_last = sample_n

    return phase_changes_sample_n, phase_changes_volume


class AudioHandler:
    def __init__(self, settings_handler):
        """
        Initializes AudioHandler class
        :param settings_handler:
        """
        self.settings_handler = settings_handler

        # Streams
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
        self.audio_latency_samples = -1
        self.label_latency_update_signal = None
        self.update_label_info = None
        self.measurement_timer_start_signal = None
        self.chunk_size = 0
        self.stop_flag = False

    def calculate_chunk_size(self, sample_rate):
        """
        Calculates chunk size by sample rate
        :param sample_rate:
        :return:
        """
        self.chunk_size = sample_rate_to_chunk_size(sample_rate)

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
        self.calculate_chunk_size(sample_rate)

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
                devices_names.append(fix_encoding(
                    self.py_audio.get_device_info_by_host_api_device_index(0, i).get('name')))

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
                if device_name.lower() in str(fix_encoding(device.get('name'))).lower():
                    return device.get('index')

        except Exception as e:
            print(e)
            traceback.print_exc()

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
        threading.Thread(target=self.measure_latency_thread).start()

    def measure_latency_thread(self):
        """
        Measures latency several times to check result
        :return:
        """
        try:
            # Clear stop flag
            self.stop_flag = False

            # Reset latency
            self.audio_latency_samples = -1

            # Sample rate
            sample_rate = int(self.settings_handler.settings['audio_sample_rate'])

            # Number of channels
            recording_channels = int(self.settings_handler.settings['audio_recording_channels'])

            # FFT window
            window_type = int(self.settings_handler.settings['fft_window_type'])

            # Start latency measurement loop
            self.measure_latency_thread_running = True
            latency_samples = self.measure_latency_loop(sample_rate, recording_channels, window_type)
            if latency_samples < 0 or self.stop_flag:
                self.audio_latency_samples = -1
            else:
                self.audio_latency_samples = latency_samples

            # Display measured latency
            if self.label_latency_update_signal is not None:
                if self.audio_latency_samples >= 0:
                    self.label_latency_update_signal.emit('Latency: ' +
                                                          str(self.audio_latency_samples) + ' samples ('
                                                          + str(round(self.audio_latency_samples
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

    def measure_latency_loop(self, sample_rate, recording_channels, window_type):
        """
        Measures latency (in samples) between playback and recording
        TODO: Improve latency measurement
        :param sample_rate:
        :param recording_channels:
        :param window_type:
        :return:
        """
        # Loop variables
        latency_samples = -1
        chunk_counter = 0

        # Recording buffer to store all samples for future analysis
        recording_buffer = np.empty(0, dtype=np.float32)

        # Frequency
        test_frequency_samples = self.chunk_size / 16

        # Generate samples:
        # chunk_size * 0 - 1: test_frequency_samples
        # phase change
        # chunk_size * 2 - 3: test_frequency_samples
        # phase change
        # chunk_size * 4 - 5: test_frequency_samples
        samples_buffer = np.empty(0, dtype=np.float32)
        samples_buffer_silence = np.zeros(self.chunk_size, dtype=np.float32)
        phase = 0
        for sample_chunk_n in range(6):
            # Invert phase 2 times
            if sample_chunk_n == 2 or sample_chunk_n == 4:
                phase += np.pi

            # Fill buffer with sine wave
            for sample in range(self.chunk_size):
                sample = MEASURE_LATENCY_VOLUME * np.sin(phase)
                phase += 2 * np.pi / test_frequency_samples
                samples_buffer = np.append(samples_buffer, [sample])

        # Generate window
        window = generate_window(window_type, self.chunk_size)

        # Play silence
        self.play_silence()

        signal_receiving_start = False
        while self.measure_latency_thread_running:
            # Convert to bytes
            if chunk_counter * self.chunk_size < len(samples_buffer) - 2:
                output_bytes = array.array('f', samples_buffer[self.chunk_size * chunk_counter:
                                                               self.chunk_size * (chunk_counter + 1)]).tobytes()
            else:
                output_bytes = array.array('f', samples_buffer_silence).tobytes()

            # Write to stream
            self.playback_stream.write(output_bytes)

            # Read data
            input_data_raw = self.recording_stream.read(self.chunk_size)
            input_data = np.frombuffer(input_data_raw, dtype=np.float3232)

            # Split into channels and make mono
            input_data = input_data.reshape((len(input_data) // recording_channels, recording_channels))
            data_per_channels = np.split(input_data, recording_channels, axis=1)
            input_data_mono = data_per_channels[0].flatten()
            for channel_n in range(1, recording_channels):
                input_data_mono = np.add(input_data_mono, data_per_channels[channel_n].flatten())
            input_data_mono = np.divide(input_data_mono, recording_channels)

            # Append to recording buffer
            recording_buffer = np.append(recording_buffer, input_data_mono)

            # Compute FFT
            fft_dbfs = compute_fft_dbfs(input_data_mono, window, window_type)

            # Mean of signal (dbfs)
            fft_mean = np.mean(fft_dbfs)

            # Real peak value and index
            fft_max = np.max(fft_dbfs)
            fft_max_index = np.where(fft_dbfs == fft_max)[0][0]
            fft_max_frequency_hz = index_to_frequency(fft_max_index, sample_rate, self.chunk_size)
            expected_frequency_hz = sample_rate / test_frequency_samples

            # Print info
            if self.update_label_info is not None:
                self.update_label_info.emit('Mean level: ' + str(int(fft_mean)) + ' dBFS, Peak level: '
                                            + str(int(fft_max)) + ' dBFS, Expected f: '
                                            + str(int(fft_max_frequency_hz)) + ' Hz')

            # Detect that recording started
            if fft_max / fft_mean < 0.5 \
                    and abs(expected_frequency_hz - fft_max_frequency_hz) < MEASURE_LATENCY_ACCEPTED_DEVIATION_HZ:
                if not signal_receiving_start:
                    signal_receiving_start = True

            # Exit if no more signal
            elif signal_receiving_start:
                self.measure_latency_thread_running = False
                break

            # Count chunks
            chunk_counter += 1

            # Timeout error measuring latency
            if chunk_counter >= MEASURE_LATENCY_MAX_LATENCY_CHUNKS:
                self.error_message = 'Timeout error (no signal detected in ' \
                                     + str(MEASURE_LATENCY_MAX_LATENCY_CHUNKS) + ' chunks)'
                latency_samples = -1
                self.measure_latency_thread_running = False
                break

        # If exited successfully
        if self.error_message == '':
            # Measure peak volume
            volume = (abs(np.min(recording_buffer)) + abs(np.max(recording_buffer))) / 2

            # Check volume
            if volume < MEASURE_LATENCY_MIN_VOLUME:
                self.error_message = 'Volume too low'
            elif volume > MEASURE_LATENCY_MAX_VOLUME:
                self.error_message = 'Volume too high'
            else:
                # Find signal peaks
                peaks_positive, _ = find_peaks(recording_buffer,
                                               height=(volume / MEASURE_LATENCY_PEAKS_THRESHOLD_VOLUME, volume * 2))
                peaks_negative, _ = find_peaks(-recording_buffer,
                                               height=(volume / MEASURE_LATENCY_PEAKS_THRESHOLD_VOLUME, volume * 2))

                # Find phase changes
                phase_changes_positive_sample_n, phase_changes_positive_lvl \
                    = find_phase_changes_by_peaks(recording_buffer, peaks_positive, test_frequency_samples, True)
                phase_changes_negative_sample_n, phase_changes_negative_lvl \
                    = find_phase_changes_by_peaks(recording_buffer, peaks_negative, test_frequency_samples, False)

                # Check if there is both phase changes
                if len(phase_changes_positive_sample_n) > 0 and len(phase_changes_negative_sample_n):
                    # Find latencies by lowest volume in gap
                    latency_positive = phase_changes_positive_sample_n[
                        np.where(phase_changes_positive_lvl == np.min(phase_changes_positive_lvl))[0][0]]
                    latency_negative = phase_changes_negative_sample_n[
                        np.where(phase_changes_negative_lvl == np.min(phase_changes_negative_lvl))[0][0]]

                    # Phase independent latencies (one is chunk_size * 2 off from other)
                    latency_lower = min(latency_positive, latency_negative)
                    latency_upper = max(latency_positive, latency_negative)

                    # Subtract chunk_size * 2 from largest latency
                    latency_upper -= self.chunk_size * 2

                    # Subtract waves buffer from both latencies. After that we have real latencies
                    latency_lower -= self.chunk_size * 2
                    latency_upper -= self.chunk_size * 2

                    # Finally, check them
                    if abs(latency_lower - latency_upper) <= MEASURE_LATENCY_MAX_TOLERANCE_SAMPLES:
                        # Calculate final latency
                        latency_samples = int((latency_lower + latency_upper) / 2)

                        # Check sign
                        if latency_samples < 0:
                            latency_samples = -1
                            self.error_message = 'Measured latency is negative'
                    else:
                        self.error_message = 'Measured latencies are not equal. Distortion or low sound level'
                else:
                    self.error_message = 'Cannot detect both phase changes. The signal may be distorted'

        return latency_samples

    def play_silence(self):
        # Play silence
        if self.update_label_info is not None:
            self.update_label_info.emit('Playing silence for ' + str(SILENCE_BEFORE_MEASUREMENT_CHUNKS) + ' chunks...')
        samples_buffer_silence = np.zeros(self.chunk_size, dtype=np.float32)
        for _ in range(SILENCE_BEFORE_MEASUREMENT_CHUNKS):
            output_bytes = array.array('f', samples_buffer_silence).tobytes()
            self.playback_stream.write(output_bytes)
            self.recording_stream.read(self.chunk_size)

    def stop_measuring_latency(self):
        """
        Stops measuring latency
        :return:
        """
        # Set stop flag
        self.stop_flag = True

        if self.measure_latency_thread_running:
            # Stop loop
            self.measure_latency_thread_running = False

            # Reset latency
            self.audio_latency_samples = -1

            # Display message
            if self.label_latency_update_signal is not None:
                self.label_latency_update_signal.emit('Latency measurement stopped!')
