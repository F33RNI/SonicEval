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
import csv
import datetime

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from pyqtgraph import exporters


def export_image(dialog_parent, graph_widget):
    """
    Exports plot as png image
    :return:
    """
    try:
        # Timestamp for filename
        timestamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

        # Ask for filename
        file_name, _ = QFileDialog.getSaveFileName(dialog_parent,
                                                   'Save file', 'frequency_response_' + timestamp + '.png')

        if file_name is not None and len(file_name) > 0:
            # Check extension
            if not file_name.endswith('.png'):
                file_name += '.png'

            # Export as image
            exporter = exporters.ImageExporter(graph_widget.scene())
            exporter.export(file_name)
            QMessageBox.information(dialog_parent, 'Image saved', 'Image saved to:\n' + file_name, QMessageBox.Ok)

    # Error
    except Exception as e:
        QMessageBox.critical(dialog_parent, 'Error', 'Error exporting image!\n\n' + str(e), QMessageBox.Ok)


def export_data_as_csv(csv_writer, frequency_list, data_dbfs):
    """
    Writes data to csv_writer
    :param csv_writer:
    :param data_dbfs: [[ch1_point_1, ch1_point_2,...], [ch2_point_1, ch2_point_2,...]]
    :param frequency_list: [1, 10,...]
    :return:
    """
    # Transpose array so each arrow is one point (all channels)
    data_dbfs = data_dbfs.transpose()

    # List all points
    for i in range(len(data_dbfs)):
        # Create and write one row
        csv_row = [str(int(frequency_list[i]))]
        for channel_n in range(len(data_dbfs[0])):
            csv_row.append(str(data_dbfs[i][channel_n]).replace(',', '.'))
        csv_writer.writerow(csv_row)


class FileLoader:
    def __init__(self, settings_handler):
        self.settings_handler = settings_handler

    def load_from_file(self, dialog_parent):
        """
        Load frequency response from file
        :param dialog_parent: parent
        :return: recording_channels, frequencies, data_dbfs
        """
        # Result
        recording_channels = 0
        frequencies = []
        data_dbfs = []

        # Create open file dialog
        file_dialog = QFileDialog(dialog_parent)
        file_dialog.setWindowTitle('Open file')
        file_dialog.setNameFilters(['CSV Files (*.csv)', 'All Files (*)'])
        file_dialog.selectNameFilter('CSV Files (*.csv)')
        file_dialog.exec_()

        # Ask for file
        selected_files = file_dialog.selectedFiles()

        # Check file name
        if selected_files is not None and len(selected_files) > 0:
            file_name = selected_files[0]
            if file_name is not None and len(file_name) > 0:
                # Open file
                file = open(file_name, newline='')
                csv_reader = csv.reader(file, delimiter=str(self.settings_handler.settings['csv_separator']))

                # First time flag
                initialized = False

                # Parse data
                for row in csv_reader:
                    # First time reading (header)
                    if not initialized:
                        recording_channels = len(row) - 1
                        initialized = True

                    else:
                        # Frequency
                        frequencies.append(int(float(row[0])))

                        # Parse levels
                        levels = []
                        for channel_level in row[1:]:
                            # Convert to float
                            channel_level = float(channel_level.replace(',', '.').strip())
                            levels.append(channel_level)

                        # Levels
                        data_dbfs.append(levels)

                # Transpose array
                data_dbfs = np.array(data_dbfs, dtype=float).transpose()

        return recording_channels, frequencies, data_dbfs

    def save_to_file(self, dialog_parent, frequencies, levels):
        """
        Saves data to csv file
        :param dialog_parent: parent
        :param frequencies: [f1, f2,...]
        :param levels: [[ch1_level1, ch1_level2,...], [ch2_level1, ch2_level2,...]]
        :return:
        """
        # Timestamp for filename
        timestamp = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

        # Ask for filename
        file_name, _ = QFileDialog.getSaveFileName(dialog_parent, 'Save File',
                                                   'frequency_response_' + timestamp + '.csv')
        if file_name is not None and len(file_name) > 0:
            # Check extension
            if not file_name.endswith('.csv'):
                file_name += '.csv'

            # Create CSV writer
            file = open(file_name, 'w', encoding='UTF8', newline='')
            csv_writer = csv.writer(file, delimiter=str(self.settings_handler.settings['csv_separator']))

            # CSV header
            csv_header = ['Frequency (Hz)']
            for channel_n in range(len(levels)):
                csv_header.append('Channel ' + str(channel_n + 1) + ' level (dBFS)')
            csv_writer.writerow(csv_header)

            # Export data
            export_data_as_csv(csv_writer, frequencies, levels)

            # Close file
            file.close()

            # Show file path
            QMessageBox.information(dialog_parent, 'Data saved', 'Frequency response saved to:\n' + file_name,
                                    QMessageBox.Ok)
