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
import ctypes
import os
import sys

import psutil
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

import AudioHandler
import FileLoader
import GraphPlot
import NoiseHandler
import SettingsHandler
import SweepHandler

APP_VERSION = '1.0.0'

SETTINGS_FILE = 'settings.json'

UPDATE_SETTINGS_AFTER_MS = 200

BTN_MEASUREMENT_START_TEXT = 'Start measurement'
BTN_MEASUREMENT_STOP_TEXT = 'Stop'

# Defines
BTN_MEASUREMENT_ACTION_START = 0
BTN_MEASUREMENT_ACTION_STOP = 1
MEASUREMENT_STAGE_IDLE = 0
MEASUREMENT_STAGE_LATENCY = 1
MEASUREMENT_STAGE_FREQUENCY_RESPONSE = 2


class Window(QMainWindow):
    update_label_latency = QtCore.pyqtSignal(str)  # QtCore.Signal(str)
    update_label_info = QtCore.pyqtSignal(str)  # QtCore.Signal(str)
    update_measurement_progress = QtCore.pyqtSignal(int)  # QtCore.Signal(int)
    measurement_continue_timer_start_signal = QtCore.pyqtSignal(int)  # QtCore.Signal(int)
    measurement_finish_timer_start_signal = QtCore.pyqtSignal(int)  # QtCore.Signal(int)
    plot_on_graph_signal = QtCore.pyqtSignal()  # QtCore.Signal()

    def __init__(self):
        super(Window, self).__init__()

        # Timer for write new settings
        self.settings_timer = QTimer()

        # Load GUI from file
        uic.loadUi('gui.ui', self)

        # Set window and app title
        self.setWindowTitle('Pulsely ' + APP_VERSION)
        self.app_title.setText('Pulsely ' + APP_VERSION)

        # Set icon
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        # Show GUI
        self.show()

        # Initialize settings class
        self.settings_handler = SettingsHandler.SettingsHandler(SETTINGS_FILE)

        # Connect buttons
        self.btn_interfaces_refresh.clicked.connect(self.interfaces_refresh)
        self.btn_measurement_start.clicked.connect(self.measurement_start)
        self.btn_graph_home.clicked.connect(self.plot_set_axes_range)
        self.btn_save_to_file.clicked.connect(self.save_to_file)
        self.btn_export_image.clicked.connect(self.export_image)
        self.btn_load_from_file.clicked.connect(self.load_from_file)
        self.cbox_normalize.clicked.connect(self.normalize_cbox_clicked)
        self.btn_load_reference.clicked.connect(self.load_reference)
        self.btn_clear_reference.clicked.connect(self.clear_reference)
        self.cbox_normalize_reference.clicked.connect(self.normalize_reference)

        # State of main button
        self.btn_measurement_action = BTN_MEASUREMENT_ACTION_START

        # Current measurement stage
        self.measurement_stage = MEASUREMENT_STAGE_IDLE

        # Measurement timers
        self.measurement_continue_timer = QtCore.QTimer()
        self.measurement_finish_timer = QtCore.QTimer()
        self.measurement_continue_timer.timeout.connect(self.measurement_continue)
        self.measurement_finish_timer.timeout.connect(self.measurement_finish)

        # Connect signals
        self.update_label_latency.connect(self.label_latency.setText)
        self.update_label_info.connect(self.label_info.setText)
        self.update_measurement_progress.connect(self.measurement_progress.setValue)
        self.measurement_continue_timer_start_signal.connect(self.measurement_continue_timer.start)
        self.measurement_finish_timer_start_signal.connect(self.measurement_finish_timer.start)
        self.plot_on_graph_signal.connect(self.plot_data)

        # Initialize window combobox
        self.fft_window_type.addItems(['None', 'Hamming', 'Hanning', 'Blackman'])

        # Parse settings
        self.settings_handler.read_from_file()

        # Initialize other classes
        self.audio_handler = AudioHandler.AudioHandler(self.settings_handler)
        self.sweep_handler = SweepHandler.SweepHandler(self.settings_handler, self.audio_handler)
        self.noise_handler = NoiseHandler.NoiseHandler(self.settings_handler, self.audio_handler)
        self.file_loader = FileLoader.FileLoader(self.settings_handler)
        self.graph_plot_main = GraphPlot.GraphPlot(self.graphWidget, self.settings_handler)
        self.graph_plot_reference = GraphPlot.GraphPlot(self.graphWidget_2, self.settings_handler)

        # Update GUI
        self.show_settings()
        self.write_settings()

        # Connect settings updater
        self.audio_playback_interface.currentTextChanged.connect(self.update_settings)
        self.audio_recording_interface.currentTextChanged.connect(self.update_settings)
        self.audio_recording_channels.valueChanged.connect(self.update_settings)
        self.audio_sample_rate.valueChanged.connect(self.update_settings)
        self.rbtn_sweep.clicked.connect(self.write_settings)
        self.rbtn_noise.clicked.connect(self.write_settings)
        self.signal_start_freq.valueChanged.connect(self.update_settings)
        self.signal_stop_freq.valueChanged.connect(self.update_settings)
        self.signal_test_duration.valueChanged.connect(self.update_settings)
        self.audio_playback_volume.valueChanged.connect(self.update_settings)
        self.fft_size_chunks.valueChanged.connect(self.update_settings)
        self.fft_window_type.currentTextChanged.connect(self.update_settings)
        self.noise_filter_order.valueChanged.connect(self.update_settings)
        self.noise_random_seed.valueChanged.connect(self.update_settings)
        self.line_csv_separator.textChanged.connect(self.update_settings)

        # Connect timer
        self.settings_timer.timeout.connect(self.write_settings)

    def measurement_start(self):
        """
        Starts / stops measurement
        :return:
        """
        # Start button pressed
        if self.btn_measurement_action == BTN_MEASUREMENT_ACTION_START:
            # Clear latency label
            self.label_latency.setText('Latency: -')

            # Disable controls
            self.disable_controls()
            self.btn_measurement_start.setText(BTN_MEASUREMENT_STOP_TEXT)
            self.btn_measurement_action = BTN_MEASUREMENT_ACTION_STOP

            # Reset progress bar
            self.measurement_progress.setValue(0)

            # Measure latency with tone
            self.measurement_stage = MEASUREMENT_STAGE_LATENCY
            self.audio_handler.measure_latency(self.update_label_latency,
                                               self.update_label_info,
                                               self.measurement_continue_timer_start_signal)

        # Stop button pressed
        elif self.btn_measurement_action == BTN_MEASUREMENT_ACTION_STOP:
            # Stop latency measurement
            self.audio_handler.stop_measuring_latency()

            # Stop frequency response measurement
            if int(self.settings_handler.settings['signal_type']) == AudioHandler.TEST_SIGNAL_TYPE_SWEEP:
                self.sweep_handler.stop_measurement()
            else:
                self.noise_handler.stop_measurement()

            # Stop all timers
            self.measurement_continue_timer.stop()
            self.measurement_finish_timer.stop()

            # Go to final step
            self.measurement_finish()

    def measurement_continue(self):
        """
        Step 2 - measure frequency responce
        :return:
        """
        # Stop timer
        self.measurement_continue_timer.stop()

        # Clear info label
        self.label_info.setText('')

        # Fatal error during latency measurement
        if len(self.audio_handler.error_message) > 0:
            # Show fatal error message
            QMessageBox.critical(self, 'Fatal error',
                                 'Fatal error during latency measurement!\nThe app will be closed!\n\n'
                                 + str(self.audio_handler.error_message), QMessageBox.Ok)

            # Kill all threads and exit
            current_system_pid = os.getpid()
            psutil.Process(current_system_pid).terminate()
            QCoreApplication.exit(-1)

        # Check latency
        if self.audio_handler.audio_latency_chunks >= 0:
            # Generate new plots
            self.graph_plot_main.plots_prepare(int(self.settings_handler.settings['audio_recording_channels']))

            # Start measurement
            if int(self.settings_handler.settings['signal_type']) == AudioHandler.TEST_SIGNAL_TYPE_SWEEP:
                self.sweep_handler.start_measurement(self.update_label_info, self.update_measurement_progress,
                                                     self.measurement_finish_timer_start_signal,
                                                     self.plot_on_graph_signal)
            else:
                self.noise_handler.start_measurement(self.update_label_info, self.update_measurement_progress,
                                                     self.measurement_finish_timer_start_signal,
                                                     self.plot_on_graph_signal)

        # Error
        else:
            # Error message
            QMessageBox.critical(self, 'Error', 'Failed to measure latency!\nCheck settings and try again')

            # Enable controls
            self.enable_controls()
            self.btn_measurement_start.setText(BTN_MEASUREMENT_START_TEXT)
            self.btn_measurement_action = BTN_MEASUREMENT_ACTION_START

    def measurement_finish(self):
        """
        Measurement final stage
        :return:
        """
        # Stop timer
        self.measurement_finish_timer.stop()

        # Clear info label
        self.label_info.setText('')

        # Reset progress bar
        self.measurement_progress.setValue(0)

        # Get error message
        if int(self.settings_handler.settings['signal_type']) == AudioHandler.TEST_SIGNAL_TYPE_SWEEP:
            error_message = self.sweep_handler.error_message
        else:
            error_message = self.noise_handler.error_message

        # Fatal error during frequency response measurement
        if len(error_message) > 0:
            # Show fatal error message
            QMessageBox.critical(self, 'Fatal error',
                                 'Fatal error during frequency response measurement!\nThe app will be closed!\n\n'
                                 + error_message, QMessageBox.Ok)

            # Kill all threads and exit
            current_system_pid = os.getpid()
            psutil.Process(current_system_pid).terminate()
            QCoreApplication.exit(-1)

        else:
            # Enable controls
            self.enable_controls()
            self.btn_measurement_start.setText(BTN_MEASUREMENT_START_TEXT)

            # Final message
            signal_type = int(self.settings_handler.settings['signal_type'])
            if (signal_type == AudioHandler.TEST_SIGNAL_TYPE_SWEEP and self.sweep_handler.measurement_completed) or \
                    (signal_type == AudioHandler.TEST_SIGNAL_TYPE_NOISE and self.noise_handler.measurement_completed):
                QMessageBox.information(self, 'Done', 'Frequency response measurement completed!', QMessageBox.Ok)

            # Restore default button action
            self.btn_measurement_action = BTN_MEASUREMENT_ACTION_START

    def load_reference(self):
        """
        Loads reference frequency response
        :return:
        """
        try:
            # Load from file
            recording_channels, frequencies, data_dbfs = self.file_loader.load_from_file(self)
            if recording_channels > 0 and len(frequencies) > 0 and len(data_dbfs) > 0:
                # Assign internal variables
                self.audio_handler.reference_frequencies = frequencies.copy()
                self.audio_handler.reference_levels_per_channels = data_dbfs.copy()

                # Create plots
                self.graph_plot_reference.plots_prepare(len(self.audio_handler.reference_levels_per_channels))

                # Plot graph
                self.graph_plot_reference.plot_data(frequencies, data_dbfs, self.cbox_normalize_reference.isChecked())

        # Error
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Error loading file!\n\n' + str(e), QMessageBox.Ok)

        # Re-plot main graph
        self.plot_data()

    def clear_reference(self):
        """
        Clears reference frequency response
        :return:
        """
        self.audio_handler.reference_frequencies = []
        self.audio_handler.reference_levels_per_channels = []
        self.graph_plot_reference.plot_clear()

        # Re-plot main graph
        self.plot_data()

    def normalize_reference(self):
        """
        cbox_normalize_reference clicked
        :return:
        """
        # Replot current data
        if self.audio_handler.reference_frequencies is not None \
                and len(self.audio_handler.reference_frequencies) > 0:
            self.graph_plot_reference.plot_data(self.audio_handler.reference_frequencies,
                                                self.audio_handler.reference_levels_per_channels,
                                                self.cbox_normalize_reference.isChecked())

        # Re-plot main graph
        self.plot_data()

        # Update settings
        self.write_settings()

    def show_settings(self):
        """
        Updates gui elements from settings
        :return:
        """
        # Audio interface
        self.interfaces_refresh()
        self.audio_recording_channels.setValue(int(self.settings_handler.settings['audio_recording_channels']))
        self.audio_sample_rate.setValue(int(self.settings_handler.settings['audio_sample_rate']))
        self.label_chunk_size.setText(str(AudioHandler.sample_rate_to_chunk_size(
            int(self.settings_handler.settings['audio_sample_rate']))))

        # Test signal
        self.rbtn_sweep.setChecked(True if int(self.settings_handler.settings['signal_type']) == AudioHandler.
                                   TEST_SIGNAL_TYPE_SWEEP else False)
        self.rbtn_noise.setChecked(True if int(self.settings_handler.settings['signal_type']) == AudioHandler.
                                   TEST_SIGNAL_TYPE_NOISE else False)
        self.signal_start_freq.setValue(int(self.settings_handler.settings['signal_start_freq']))
        self.signal_start_freq.setMaximum(int(self.settings_handler.settings['audio_sample_rate']) // 2 - 1)
        self.signal_stop_freq.setValue(int(self.settings_handler.settings['signal_stop_freq']))
        self.signal_stop_freq.setMaximum(int(self.settings_handler.settings['audio_sample_rate']) // 2 - 1)
        self.signal_test_duration.setValue(int(self.settings_handler.settings['signal_test_duration']))
        self.audio_playback_volume.setValue(int(self.settings_handler.settings['audio_playback_volume']))

        # FFT
        self.fft_size_chunks.setValue(int(self.settings_handler.settings['fft_size_chunks']))
        self.fft_window_type.setCurrentIndex(int(self.settings_handler.settings['fft_window_type']))

        # Noise
        self.noise_filter_order.setValue(int(self.settings_handler.settings['noise_filter_order']))
        self.noise_random_seed.setValue(int(self.settings_handler.settings['noise_random_seed']))

        # File
        self.line_csv_separator.setText(str(self.settings_handler.settings['csv_separator']))

        # Normalize checkboxes
        self.cbox_normalize.setChecked(self.settings_handler.settings['normalize_frequency_response'])
        self.cbox_normalize_reference.setChecked(self.settings_handler.settings['normalize_reference'])

        # Update charts
        self.graph_plot_main.init_chart(self.audio_handler.frequency_response_frequencies)
        self.graph_plot_reference.init_chart(self.audio_handler.reference_frequencies, False)

    def interfaces_refresh(self):
        """
        Shows list on playback and recording audio devices
        :return:
        """
        # Re-initialise PyAudio
        self.audio_handler.initialize_py_audio()

        # Audio playback interface
        self.audio_playback_interface.clear()
        playback_devices = self.audio_handler.get_devices_names(AudioHandler.DEVICE_TYPE_OUTPUT)
        for playback_device in playback_devices:
            self.audio_playback_interface.addItem(str(playback_device))

        # Select playback interface
        playback_device = str(self.settings_handler.settings['audio_playback_interface'])
        if playback_device in playback_devices:
            self.audio_playback_interface.setCurrentText(playback_device)

        # Audio recording interface
        self.audio_recording_interface.clear()
        recording_devices = self.audio_handler.get_devices_names(AudioHandler.DEVICE_TYPE_INPUT)
        for recording_device in recording_devices:
            self.audio_recording_interface.addItem(str(recording_device))

        # Select recording interface
        recording_device = str(self.settings_handler.settings['audio_recording_interface'])
        if recording_device in recording_devices:
            self.audio_recording_interface.setCurrentText(recording_device)

    def update_settings(self):
        """
        Starts timer to write new settings
        :return:
        """
        # Start timer
        self.settings_timer.start(UPDATE_SETTINGS_AFTER_MS)

    def write_settings(self):
        """
        Writes new settings to file
        :return:
        """
        # Stop timer
        self.settings_timer.stop()

        # Audio interface
        self.settings_handler.settings['audio_playback_interface'] = str(self.audio_playback_interface.currentText())
        self.settings_handler.settings['audio_recording_interface'] = str(self.audio_recording_interface.currentText())
        self.settings_handler.settings['audio_recording_channels'] = int(self.audio_recording_channels.value())
        self.settings_handler.settings['audio_sample_rate'] = int(self.audio_sample_rate.value())
        self.label_chunk_size.setText(str(AudioHandler.sample_rate_to_chunk_size(
            int(self.settings_handler.settings['audio_sample_rate']))))

        # Test signal
        self.settings_handler.settings['signal_type'] = AudioHandler.TEST_SIGNAL_TYPE_SWEEP \
            if self.rbtn_sweep.isChecked() else AudioHandler.TEST_SIGNAL_TYPE_NOISE
        self.settings_handler.settings['signal_start_freq'] = int(self.signal_start_freq.value())
        self.signal_start_freq.setMaximum(int(self.settings_handler.settings['audio_sample_rate']) // 2 - 1)
        self.settings_handler.settings['signal_stop_freq'] = int(self.signal_stop_freq.value())
        self.signal_stop_freq.setMaximum(int(self.settings_handler.settings['audio_sample_rate']) // 2 - 1)
        self.settings_handler.settings['signal_test_duration'] = int(self.signal_test_duration.value())
        self.settings_handler.settings['audio_playback_volume'] = int(self.audio_playback_volume.value())

        # FFT
        self.settings_handler.settings['fft_size_chunks'] = int(self.fft_size_chunks.value())
        self.settings_handler.settings['fft_window_type'] = int(self.fft_window_type.currentIndex())

        # Noise
        self.settings_handler.settings['noise_filter_order'] = int(self.noise_filter_order.value())
        self.settings_handler.settings['noise_random_seed'] = int(self.noise_random_seed.value())

        # File
        self.settings_handler.settings['csv_separator'] = str(self.line_csv_separator.text())

        # Normalize checkboxes
        self.settings_handler.settings['normalize_frequency_response'] = self.cbox_normalize.isChecked()
        self.settings_handler.settings['normalize_reference'] = self.cbox_normalize_reference.isChecked()

        # Write new settings to file
        self.settings_handler.write_to_file()

    def plot_set_axes_range(self):
        """
        Set axes range of plot widget
        :return:
        """
        self.graph_plot_main.plot_set_axes_range(self.audio_handler.frequency_response_frequencies)

    def normalize_cbox_clicked(self):
        """
        Rep-lots data and writes settings
        :return:
        """
        # Replot current data
        if self.audio_handler.frequency_response_frequencies is not None \
                and len(self.audio_handler.frequency_response_frequencies) > 0:
            self.plot_data()

        # Update settings
        self.write_settings()

    def plot_data(self):
        """
        Plots data on graph
        :return:
        """
        # Unpack data
        frequencies = self.audio_handler.frequency_response_frequencies.copy()
        levels = self.audio_handler.frequency_response_levels_per_channels.copy()

        # Plot data
        self.graph_plot_main.plot_data(frequencies, levels, self.cbox_normalize.isChecked(),
                                       self.audio_handler.reference_frequencies,
                                       self.audio_handler.reference_levels_per_channels,
                                       self.cbox_normalize_reference.isChecked())

    def save_to_file(self):
        """
        Saves frequency response to csv file
        :return:
        """
        try:
            # Unpack data
            frequencies = self.audio_handler.frequency_response_frequencies.copy()
            levels = self.audio_handler.frequency_response_levels_per_channels.copy()

            if levels is not None and len(levels) > 0:
                # Apply reference?
                ref_frequencies = self.audio_handler.reference_frequencies
                ref_levels = self.audio_handler.reference_levels_per_channels
                if ref_frequencies is not None and ref_levels is not None \
                        and len(ref_frequencies) > 0 and len(ref_levels) > 0:
                    levels = AudioHandler.apply_reference(frequencies, levels, ref_frequencies, ref_levels,
                                                          self.cbox_normalize_reference.isChecked())

                # Normalize?
                if self.cbox_normalize.isChecked():
                    levels = AudioHandler.normalize_data(levels)

                # Save to file
                self.file_loader.save_to_file(self, frequencies, levels)

            # No data yet
            else:
                QMessageBox.warning(self, 'No data', 'No data to save!', QMessageBox.Ok)

        # Error
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Error saving file!\n\n' + str(e), QMessageBox.Ok)

    def export_image(self):
        FileLoader.export_image(self, self.graphWidget)

    def load_from_file(self):
        """
        Loads data from csv file
        :return:
        """
        try:
            # Load from file
            recording_channels, frequencies, data_dbfs = self.file_loader.load_from_file(self)
            if recording_channels > 0 and len(frequencies) > 0 and len(data_dbfs) > 0:
                # Assign internal variables
                self.audio_handler.frequency_response_frequencies = frequencies.copy()
                self.audio_handler.frequency_response_levels_per_channels = data_dbfs.copy()

                # Create plots
                self.graph_plot_main.plots_prepare(len(self.audio_handler.frequency_response_levels_per_channels))

                # Plot graph
                self.plot_data()

        # Error
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Error loading file!\n\n' + str(e), QMessageBox.Ok)

    def disable_controls(self):
        """
        Disables controls (to prevent settings change during measurement)
        :return:
        """
        self.settings_area.setEnabled(False)
        self.btn_export_image.setEnabled(False)

    def enable_controls(self):
        """
        Enables controls (to prevent settings change during measurement)
        :return:
        """
        self.settings_area.setEnabled(True)
        self.btn_export_image.setEnabled(True)

    def closeEvent(self, event):
        """
        Asks for exit confirmation
        :param event:
        :return:
        """
        # Close confirmation
        quit_msg = 'Are you sure you want to exit?'
        reply = QMessageBox.warning(self, 'Exit confirmation', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Kill all threads
            current_system_pid = os.getpid()
            psutil.Process(current_system_pid).terminate()
            event.accept()

        # Stay in app
        else:
            event.ignore()


if __name__ == '__main__':
    # Replace icon in taskbar
    pulsely_app_ip = 'f3rni.pulsely.pulsely.' + APP_VERSION
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(pulsely_app_ip)

    # Start app
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('fusion')
    win = Window()
    app.exec_()
