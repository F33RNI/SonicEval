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
import threading
import traceback

import numpy as np
from PyQt5 import QtCore

from AudioHandler import generate_window, compute_fft_dbfs, TEST_SIGNAL_TYPE_NOISE, butter_bandpass_filter, \
    index_to_frequency, frequency_to_index, clamp

# Filter formula:
# filter_k = -(((sample_rate / chunk_size) * fft_size_chunks) / (FILTER_SCALE * signal_duration_chunks)) + 1
# filtered_data = filtered_data * filter_k + unfiltered_data * (1. - filter_k)
# less FILTER_SCALE -> faster noise settle, coarser signal
# more FILTER_SCALE -> slower noise settle, softer signal, may not have time to settle
FILTER_SCALE = 18.


class NoiseHandler:
    def __init__(self, settings_handler, audio_handler):
        """
        Initializes NoiseHandler class
        :param settings_handler:
        """
        self.settings_handler = settings_handler
        self.audio_handler = audio_handler

        self.error_message = ''

        self.noise_thread_running = False
        self.plot_on_graph_signal = None
        self.update_label_info = None
        self.update_measurement_progress = None
        self.measurement_timer_start_signal = None
        self.measurement_completed = False

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

        # Clear flag
        self.measurement_completed = False

        # Start noise frequency response as thread
        self.noise_thread_running = True
        threading.Thread(target=self.noise_loop).start()

    def noise_loop(self):
        try:
            # Open audio stream
            recording_channels = int(self.settings_handler.settings['audio_recording_channels'])
            playback_stream, recording_stream = self.audio_handler.open_audio(recording_channels)

            # Get settings and other constants
            chunk_size = self.audio_handler.chunk_size
            volume = int(self.settings_handler.settings['audio_playback_volume']) / 100.
            sample_rate = int(self.settings_handler.settings['audio_sample_rate'])
            signal_start_freq = clamp(int(self.settings_handler.settings['signal_start_freq']), 0, sample_rate // 2 - 1)
            signal_stop_freq = clamp(int(self.settings_handler.settings['signal_stop_freq']), 0, sample_rate // 2 - 1)
            signal_duration_s = int(self.settings_handler.settings['signal_test_duration'])
            fft_size_chunks = int(self.settings_handler.settings['fft_size_chunks'])
            noise_filter_order = int(self.settings_handler.settings['noise_filter_order'])
            np.random.seed(int(self.settings_handler.settings['noise_random_seed']))
            latency_samples = self.audio_handler.audio_latency_samples

            # Calculate latency buffer sizes
            latency_chunks = (latency_samples // chunk_size) + 1
            latency_samples_offset = chunk_size - int(latency_samples % chunk_size)

            # Calculate signal duration in chunks
            signal_duration_chunks = (signal_duration_s * sample_rate) // chunk_size

            # FFT window
            fft_window_type = int(self.settings_handler.settings['fft_window_type'])
            window = generate_window(fft_window_type, chunk_size * fft_size_chunks)

            # Buffer to increase delay to fil into full chunk
            input_data_offset_buffer = np.zeros((chunk_size + latency_samples_offset) * recording_channels,
                                                dtype=np.float32)

            # Recording data buffer (floats)
            fft_buffer = np.zeros(chunk_size * fft_size_chunks * recording_channels,
                                  dtype=np.float32)

            # Counters
            fft_buffer_position = 0
            latency_chunk_counter = 0
            chunk_counter = 0

            # Calculate filter
            filter_k = -(((sample_rate / chunk_size) * fft_size_chunks) / (FILTER_SCALE * signal_duration_chunks)) + 1
            if filter_k < 0.:
                filter_k = 0
            elif filter_k > 1.:
                filter_k = 1

            # X axis
            fft_frequencies = (np.arange((chunk_size * fft_size_chunks / 2) + 1)
                               / (float(chunk_size * fft_size_chunks) / sample_rate))

            # Resulted data (per channel)
            noise_result_dbfs = np.ones((recording_channels, chunk_size * fft_size_chunks // 2 + 1),
                                        dtype=np.float32) * -np.inf

            # Clear existing data
            self.audio_handler.frequency_response_frequencies = []
            self.audio_handler.frequency_response_levels_per_channels = []

            while self.noise_thread_running:
                # Generate noise from -1 to 1
                samples = np.random.random(chunk_size) * 2
                samples -= 1
                samples *= volume
                samples = butter_bandpass_filter(samples, signal_start_freq, signal_stop_freq,
                                                 sample_rate, order=noise_filter_order)

                # Convert to bytes
                output_bytes = array.array('f', samples).tobytes()

                # Write to stream
                playback_stream.write(output_bytes)

                # Read data
                input_data_raw = recording_stream.read(chunk_size, exception_on_overflow=False)
                input_data = np.frombuffer(input_data_raw, dtype=np.float32)

                # Write new data to the end of the buffer
                input_data_offset_buffer[-chunk_size * recording_channels:] = input_data

                # Move to the left,
                # so new tails will be moves to the buffer start and buffer start to the chunk start
                input_data_offset_buffer \
                    = np.roll(input_data_offset_buffer, latency_samples_offset * recording_channels)

                # Get delayed data from buffer end
                input_data_ = input_data_offset_buffer[-chunk_size * recording_channels:]

                # Delay reached
                if latency_chunk_counter >= latency_chunks:
                    # Fill measurement buffer
                    fft_buffer[fft_buffer_position:
                               fft_buffer_position + chunk_size * recording_channels] = input_data_
                    fft_buffer_position += chunk_size * recording_channels

                    # Measurement buffer is full
                    if fft_buffer_position == chunk_size * fft_size_chunks * recording_channels:
                        # Reset measurement buffer position
                        fft_buffer_position = 0

                        # Split into channels
                        input_data = fft_buffer.reshape((len(fft_buffer) // recording_channels,
                                                         recording_channels))
                        data_per_channels = np.split(input_data, recording_channels, axis=1)

                        # Info data
                        fft_peak_hz_avg = 0
                        fft_peak_dbfs_avg = 0
                        fft_mean_avg_dbfs_avg = 0

                        # Input data length
                        data_length = len(data_per_channels[0].flatten())

                        # Compute FFT for each channel
                        for channel_n in range(recording_channels):
                            # Compute FFT
                            fft_dbfs = compute_fft_dbfs(data_per_channels[channel_n].flatten(), window, fft_window_type,
                                                        TEST_SIGNAL_TYPE_NOISE)

                            # First run - initialize filter
                            if np.mean(noise_result_dbfs[channel_n]) == -np.inf:
                                noise_result_dbfs[channel_n] = fft_dbfs

                            # Filter data
                            else:
                                noise_result_dbfs[channel_n] = \
                                    noise_result_dbfs[channel_n] * filter_k + fft_dbfs * (1. - filter_k)

                            # Calculate info
                            max_peak = np.max(fft_dbfs)
                            fft_peak_hz_avg += index_to_frequency(
                                np.where(fft_dbfs == max_peak)[0][0], sample_rate, data_length)
                            fft_peak_dbfs_avg += max_peak
                            fft_mean_avg_dbfs_avg += np.mean(fft_dbfs)

                        # Calculate average info
                        fft_peak_hz_avg /= recording_channels
                        fft_peak_dbfs_avg /= recording_channels
                        fft_mean_avg_dbfs_avg /= recording_channels

                        # Cut data to bandwidth
                        bandwidth_index_start = frequency_to_index(signal_start_freq, sample_rate, data_length)
                        bandwidth_index_stop = frequency_to_index(signal_stop_freq, sample_rate, data_length)

                        # Send data to AudioHandler class
                        self.audio_handler.frequency_response_frequencies \
                            = fft_frequencies[bandwidth_index_start: bandwidth_index_stop]
                        self.audio_handler.frequency_response_levels_per_channels \
                            = noise_result_dbfs[:, bandwidth_index_start: bandwidth_index_stop]

                        # Plot graph
                        if self.plot_on_graph_signal is not None:
                            self.plot_on_graph_signal.emit()

                        # Print info
                        if self.update_label_info is not None:
                            self.update_label_info.emit('Peak: ' + str(int(fft_peak_hz_avg)) + ' Hz '
                                                        + str(int(fft_peak_dbfs_avg)) + ' dBFS, Mean lvl: '
                                                        + str(int(fft_mean_avg_dbfs_avg)) + ' dBFS')

                        # Set progress
                        if self.update_measurement_progress is not None:
                            self.update_measurement_progress.emit(int((chunk_counter / signal_duration_chunks) * 100.))

                # Wait for delay
                else:
                    # Increment delay counter
                    latency_chunk_counter += 1

                # Increment number of chunks
                chunk_counter += 1

                # Time has passed
                if chunk_counter == signal_duration_chunks:
                    # Exit
                    self.noise_thread_running = False
                    self.measurement_completed = True

            # Close audio streams
            self.audio_handler.close_audio()

            # Clear info
            if self.update_label_info is not None:
                self.update_label_info.emit('')

            # Start exit timer
            if self.measurement_timer_start_signal is not None:
                self.measurement_timer_start_signal.emit(1)

        # Error frequency response measurement
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
        if self.noise_thread_running:
            # Clear loop flag
            self.noise_thread_running = False

            # Clear info
            if self.update_label_info is not None:
                self.update_label_info.emit('')
