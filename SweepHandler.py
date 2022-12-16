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
from PyQt5 import QtCore

from AudioHandler import generate_window, compute_fft_dbfs, frequency_to_index, index_to_frequency, clamp

# How many indexes +/- to take along with the expected frequency
FREQUENCY_INDEX_TOLERANCE = 5


class SweepHandler:
    def __init__(self, settings_handler, audio_handler):
        """
        Initializes SweepHandler class
        :param settings_handler:
        """
        self.settings_handler = settings_handler
        self.audio_handler = audio_handler

        self.error_message = ''

        self.sweep_thread_running = False
        self.sweep_frequencies = []
        self.plot_on_graph_signal = None
        self.update_label_info = None
        self.update_measurement_progress = None
        self.measurement_timer_start_signal = None
        self.graph_curves = []
        self.measurement_completed = False
        self.internal_reference_dbfs = []
        self.stop_flag = False

    def map_sweep_frequencies(self):
        """
        Calculates list of frequencies to sweep from signal_start_freq to signal_stop_freq
        :return:
        """
        sample_rate = int(self.settings_handler.settings['audio_sample_rate'])

        # Calculate number of points
        one_sample_duration_s = 1. / float(sample_rate)

        chunk_duration_s = self.audio_handler.chunk_size * one_sample_duration_s
        one_point_duration_s = chunk_duration_s * int(self.settings_handler.settings['fft_size_chunks'])
        number_of_points = int(int(self.settings_handler.settings['signal_test_duration']) / one_point_duration_s)

        # Get settings
        signal_start_freq = clamp(int(self.settings_handler.settings['signal_start_freq']), 0, sample_rate // 2 - 1)
        signal_stop_freq = clamp(int(self.settings_handler.settings['signal_stop_freq']), 0, sample_rate // 2 - 1)

        self.sweep_frequencies = []
        for i in range(number_of_points):
            # Calculate sweep frequency in log scale
            sweep_frequency = (int((signal_stop_freq / 1000.) * pow(10.0, 3.0 * i / number_of_points)
                                   - signal_stop_freq / 1000. + signal_start_freq))

            # Clamp to given range (just in case)
            sweep_frequency = clamp(sweep_frequency, signal_start_freq, signal_stop_freq)

            # Append frequency to list
            self.sweep_frequencies.append(sweep_frequency)

        # Append stop frequency if not exist
        if self.sweep_frequencies[-1] != signal_stop_freq:
            self.sweep_frequencies.append(signal_stop_freq)

    def start_measurement(self, update_label_info: QtCore.pyqtSignal,
                          update_measurement_progress: QtCore.pyqtSignal,
                          measurement_timer_start_signal: QtCore.pyqtSignal,
                          plot_on_graph_signal: QtCore.pyqtSignal):
        """
        Starts sweep_loop
        :param update_label_info:
        :param update_measurement_progress:
        :param measurement_timer_start_signal:
        :param plot_on_graph_signal:
        :return:
        """
        self.update_label_info = update_label_info
        self.update_measurement_progress = update_measurement_progress
        self.measurement_timer_start_signal = measurement_timer_start_signal
        self.plot_on_graph_signal = plot_on_graph_signal

        # Calculate sweep frequencies
        self.map_sweep_frequencies()

        # Reset error message
        self.error_message = ''

        # Clear flag
        self.measurement_completed = False

        # Get settings and other constants
        chunk_size = self.audio_handler.chunk_size
        playback_device_name = str(self.settings_handler.settings['audio_playback_interface'])
        recording_device_name = str(self.settings_handler.settings['audio_recording_interface'])
        sample_rate = int(self.settings_handler.settings['audio_sample_rate'])
        volume = int(self.settings_handler.settings['audio_playback_volume']) / 100.
        recording_channels = int(self.settings_handler.settings['audio_recording_channels'])
        fft_size_chunks = int(self.settings_handler.settings['fft_size_chunks'])
        window_type = int(self.settings_handler.settings['fft_window_type'])
        latency_chunks = self.audio_handler.audio_latency_chunks

        # Clear stop flag
        self.stop_flag = False

        # Start sweep loop as thread
        self.sweep_thread_running = True
        threading.Thread(target=self.sweep_loop, args=(chunk_size, playback_device_name, recording_device_name,
                                                       sample_rate, volume, recording_channels,
                                                       fft_size_chunks, window_type, latency_chunks,)).start()

    def sweep_loop(self, chunk_size, playback_device_name, recording_device_name,
                   sample_rate, volume, recording_channels,
                   fft_size_chunks, window_type, latency_chunks, internal_calibration=False):
        """
        Measures frequency response by sweeping frequencies
        :return:
        """
        try:
            # Exit if stop flag
            if self.stop_flag:
                return

            # Perform internal calibration
            playback_stream = None
            recording_stream = None
            if not internal_calibration:
                self.sweep_loop(chunk_size, playback_device_name, recording_device_name, sample_rate, volume,
                                recording_channels, fft_size_chunks, window_type, latency_chunks, True)
                self.sweep_thread_running = True

                # Open audio streams after internal calibration
                playback_stream, recording_stream = self.audio_handler.open_audio(recording_channels)

            # Counters
            chunks_n = 0
            sweep_frequencies_position = 0
            latency_chunk_counter = 0
            input_data_buffer_position = 0

            # Delay buffer for internal_calibration mode
            samples_delay_buffer = np.zeros(chunk_size * latency_chunks, dtype=np.float)

            # FFT window
            window = generate_window(window_type, chunk_size * fft_size_chunks)

            # Buffer of frequency indexes (for delay)
            frequency_indexes_buffer = np.zeros(latency_chunks, dtype=np.int)

            # Sine wave phase
            phase = 0

            # Previous played frequency
            frequency_last = -1
            frequency_last_played_counter = 0

            # Playback data buffer (floats)
            samples = np.zeros(chunk_size, dtype=np.float)

            # Recording data buffer (floats)
            input_data_buffer = np.zeros(chunk_size * fft_size_chunks * recording_channels,
                                         dtype=np.float)

            # Resulted data (per channel)
            sweep_result_dbfs = np.empty((recording_channels, 0), dtype=np.float)
            sweep_result_frequencies = np.empty(0, dtype=int)
            result_dbfs_buffer = np.zeros(recording_channels, dtype=np.float)

            # Clear existing data
            self.audio_handler.frequency_response_frequencies = []
            self.audio_handler.frequency_response_levels_per_channels = []
            if internal_calibration:
                self.internal_reference_dbfs = []

            while self.sweep_thread_running and not self.stop_flag:
                # Current frequency
                frequency = self.sweep_frequencies[sweep_frequencies_position]

                # Rotate and fill frequency indexes buffer
                frequency_indexes_buffer = np.roll(frequency_indexes_buffer, 1)
                frequency_indexes_buffer[0] = sweep_frequencies_position

                # Fill buffer with sine wave
                for sample in range(chunk_size):
                    samples[sample] = volume * np.sin(phase)
                    phase += 2 * np.pi * frequency / sample_rate

                # Disable audio output and input if internal_calibration
                if not internal_calibration:
                    # Convert to bytes
                    output_bytes = array.array('f', samples).tobytes()

                    # Write to stream
                    playback_stream.write(output_bytes)

                    # Read data
                    input_data_raw = recording_stream.read(chunk_size)
                    input_data = np.frombuffer(input_data_raw, dtype=np.float32)

                # Pass samples to input_data with delay in internal_calibration mode (simulate latency)
                else:
                    samples_delay_buffer = np.roll(samples_delay_buffer, chunk_size)
                    samples_delay_buffer[0: chunk_size] = samples
                    input_data = np.reshape([samples_delay_buffer[len(samples_delay_buffer) - chunk_size:]]
                                            * recording_channels, chunk_size * recording_channels, order='F')

                # Delay reached
                if latency_chunk_counter >= latency_chunks - 1:
                    # Fill measurement buffer
                    input_data_buffer[input_data_buffer_position:
                                      input_data_buffer_position + chunk_size * recording_channels] \
                        = input_data
                    input_data_buffer_position += chunk_size * recording_channels

                    # Measurement buffer is full
                    if input_data_buffer_position == chunk_size * fft_size_chunks * recording_channels:
                        # Reset measurement buffer position
                        input_data_buffer_position = 0

                        # Split into channels
                        input_data = input_data_buffer.reshape((len(input_data_buffer) // recording_channels,
                                                                recording_channels))
                        data_per_channels = np.split(input_data, recording_channels, axis=1)

                        # Info data
                        fft_actual_peak_hz_avg = 0
                        fft_in_range_peak_hz_avg = 0
                        fft_mean_avg_dbfs = 0

                        # Get frequency from delay buffer
                        frequency_delayed = self.sweep_frequencies[frequency_indexes_buffer[-1]]

                        # Compute FFT for each channel
                        for channel_n in range(recording_channels):
                            # Input data length
                            data_length = len(data_per_channels[channel_n].flatten())

                            # Compute FFT
                            fft_dbfs = compute_fft_dbfs(data_per_channels[channel_n].flatten(), window, window_type)

                            # Calculate frequency indexes
                            frequency_start_index = max(
                                frequency_to_index(frequency_delayed, sample_rate, data_length)
                                - FREQUENCY_INDEX_TOLERANCE, 1)
                            frequency_stop_index = min(
                                frequency_to_index(frequency_delayed, sample_rate, data_length)
                                + FREQUENCY_INDEX_TOLERANCE, data_length // 2)

                            # Find peak value
                            peak_value = -math.inf
                            peak_index = 0
                            for frequency_index in range(frequency_start_index, frequency_stop_index):
                                if fft_dbfs[frequency_index] > peak_value:
                                    peak_value = fft_dbfs[frequency_index]
                                    peak_index = frequency_index

                            # Frequency of actual fft peak
                            fft_actual_peak_hz_avg += index_to_frequency(
                                np.where(fft_dbfs == np.max(fft_dbfs))[0][0], sample_rate, data_length)

                            # Frequency of measured peak (from frequency_start_index to frequency_stop_index)
                            fft_in_range_peak_hz_avg += index_to_frequency(peak_index, sample_rate, data_length)

                            # Mean signal level
                            fft_mean_avg_dbfs += np.mean(fft_dbfs)

                            # Add result to buffer
                            result_dbfs_buffer[channel_n] += peak_value

                            # Exit?
                            if frequency_indexes_buffer[-1] == len(self.sweep_frequencies) - 1:
                                self.sweep_thread_running = False
                                if not internal_calibration:
                                    self.measurement_completed = True

                        # Calculate average info
                        fft_actual_peak_hz_avg /= recording_channels
                        fft_in_range_peak_hz_avg /= recording_channels
                        fft_mean_avg_dbfs /= recording_channels

                        # Increment frequency change counter
                        frequency_last_played_counter += 1

                        # If frequency has changed or it was last frequency
                        if frequency_delayed != frequency_last or self.measurement_completed:
                            # Append avg level to final result
                            sweep_result_dbfs = \
                                np.append(sweep_result_dbfs,
                                          np.array([np.divide(result_dbfs_buffer,
                                                              frequency_last_played_counter)]).transpose(), axis=1)

                            # Append frequency
                            sweep_result_frequencies = np.append(sweep_result_frequencies, frequency_delayed)

                            # Clear buffer and counter
                            result_dbfs_buffer[:] = 0
                            frequency_last_played_counter = 0

                        # Store last frequency
                        frequency_last = frequency_delayed

                        # Normal mode
                        if not internal_calibration:
                            # Send level data to AudioHandler class
                            # Apply internal reference
                            if self.internal_reference_dbfs is not None and len(self.internal_reference_dbfs) > 0 \
                                    and len(self.internal_reference_dbfs[0]) >= len(sweep_result_dbfs[0]):
                                self.audio_handler.frequency_response_levels_per_channels \
                                    = np.subtract(sweep_result_dbfs,
                                                  self.internal_reference_dbfs[:, 0: len(sweep_result_dbfs[0])])
                            else:
                                self.audio_handler.frequency_response_levels_per_channels = sweep_result_dbfs

                            # Send frequency data to AudioHandler class
                            self.audio_handler.frequency_response_frequencies = sweep_result_frequencies.copy()

                            # Plot graph
                            if self.plot_on_graph_signal is not None:
                                self.plot_on_graph_signal.emit()

                            # Print info
                            if self.update_label_info is not None:
                                self.update_label_info.emit('Exp. peak: ' + str(int(frequency_delayed)) + ' Hz'
                                                            + ', Act. peak: ' + str(int(fft_actual_peak_hz_avg)) + ' Hz'
                                                            + ', Meas. peak: ' + str(int(fft_in_range_peak_hz_avg))
                                                            + ' Hz, Mean lvl: ' + str(int(fft_mean_avg_dbfs)) + ' dBFS')

                            # Set progress (2nd part, 90%)
                            if self.update_measurement_progress is not None:
                                self.update_measurement_progress.emit(
                                    int((frequency_indexes_buffer[-1] / (len(self.sweep_frequencies) - 1)) * 90.) + 10)

                        # Internal calibration
                        else:
                            # Print info
                            if self.update_label_info is not None:
                                self.update_label_info.emit('Internal calibration. Frequency: '
                                                            + str(int(frequency_delayed)) + ' Hz')

                            # Set progress (1st part, below 10%)
                            if self.update_measurement_progress is not None:
                                self.update_measurement_progress.emit(
                                    int((frequency_indexes_buffer[-1] / (len(self.sweep_frequencies) - 1)) * 10.))

                # Wait for delay
                else:
                    # Increment delay counter
                    latency_chunk_counter += 1

                # Increment chunk counter
                chunks_n += 1

                # Change frequency every CHUNKS_PER_MEASURE
                if chunks_n % fft_size_chunks == 0 and sweep_frequencies_position < len(self.sweep_frequencies) - 1:
                    # Increment sweep frequency
                    sweep_frequencies_position += 1

            # Clear info
            if self.update_label_info is not None:
                self.update_label_info.emit('')

            # Regular mode
            if not internal_calibration:
                # Close audio streams
                self.audio_handler.close_audio()

                # Start exit timer
                if self.measurement_timer_start_signal is not None:
                    self.measurement_timer_start_signal.emit(1)

            # Save internal calibration
            else:
                self.internal_reference_dbfs = sweep_result_dbfs

        # Error during sweep
        except Exception as e:
            traceback.print_exc()
            self.error_message = str(e)
            if self.measurement_timer_start_signal is not None:
                self.measurement_timer_start_signal.emit(1)

    def stop_measurement(self):
        """
        Stops current measurement
        :return:
        """
        # Set stop flag
        self.stop_flag = True

        if self.sweep_thread_running:
            # Clear loop flag
            self.sweep_thread_running = False

            # Clear info
            if self.update_label_info is not None:
                self.update_label_info.emit('')
