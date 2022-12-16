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
import math

import numpy as np
from pyqtgraph import mkPen

from AudioHandler import normalize_data, apply_reference

GRAPH_Y_RANGE_FROM = -60
GRAPH_Y_RANGE_TO = 3


class GraphPlot:
    def __init__(self, graph_widget, settings_handler):
        """
        Initializes GraphPlot class
        :param graph_widget: PyQt pyqtgraph GraphWidget
        :param settings_handler: SettingsHandler class
        """
        self.graph_widget = graph_widget
        self.settings_handler = settings_handler

        # Plots for fft
        self.graph_curves = []

    def init_chart(self, frequency_list, legend_enabled=True):
        """
        Initializes chart
        :return:
        """
        # Set fft background
        self.graph_widget.setBackground((255, 255, 255))

        # Enable grid
        self.graph_widget.showGrid(x=True, y=True, alpha=1.0)

        # Enable legend
        if legend_enabled:
            self.graph_widget.addLegend()

        # Set axes name
        self.graph_widget.setLabel('left', 'Level', units='dBFS')
        self.graph_widget.setLabel('bottom', 'Frequency', units='Hz')

        # Set axes range
        self.plot_set_axes_range(frequency_list)

        # Enable log mode on frequency scale
        self.graph_widget.setLogMode(x=True, y=False)

    def plot_set_axes_range(self, frequency_list):
        """
        Set axes range of plot widget
        :param frequency_list: list of frequencies from AudioHandler class
        :return:
        """
        # Calculate frequency range
        if frequency_list is not None and len(frequency_list) > 0:
            min_frequency = max(frequency_list[0], 1)
            max_frequency = max(frequency_list[-1], 1)
        else:
            min_frequency = 1
            max_frequency = int(self.settings_handler.settings['audio_sample_rate']) // 2

        # Set range
        self.graph_widget.setXRange(math.log10(min_frequency), math.log10(max_frequency), padding=0)
        self.graph_widget.setYRange(GRAPH_Y_RANGE_FROM, GRAPH_Y_RANGE_TO)

    def plots_prepare(self, recording_channels):
        """
        Creates new plots
        :param recording_channels:
        :return:
        """
        # Remove previous plots
        self.plot_clear()

        # Add new plots
        self.graph_curves = []
        for channel_n in range(recording_channels):
            self.graph_curves.append(self.graph_widget.plot(pen=mkPen(color=(
                int(255 - (channel_n / recording_channels) * 255),
                int((channel_n / recording_channels) * 255), 0)),
                name='Channel ' + str(channel_n + 1)))
        if recording_channels > 1:
            self.graph_curves.append(self.graph_widget.plot(pen='k', name='Sum'))

    def plot_data(self, frequencies, levels, normalize: bool,
                  ref_frequencies=None, ref_levels=None, normalize_ref=False):
        """
        Plots data on graph
        :param frequencies: list of frequencies from AudioHandler class
        :param levels: list of levels from AudioHandler class
        :param normalize: normalize data?
        :param ref_frequencies: list of frequencies from AudioHandler class
        :param ref_levels: list of levels from AudioHandler class
        :param normalize_ref: normalize reference data before applying it
        :return:
        """
        if levels is not None and len(levels) > 1:
            # Coppy levels array
            levels_copy = levels.copy()

            # Apply reference?
            if ref_frequencies is not None and ref_levels is not None \
                    and len(ref_frequencies) > 0 and len(ref_levels) > 0:
                levels_copy = apply_reference(frequencies, levels_copy, ref_frequencies, ref_levels, normalize_ref)

            # Calculate channel sum if channels > 1
            if len(levels_copy) > 1:
                levels_copy = np.vstack((levels_copy, np.average(levels_copy, axis=0)))

            # Normalize?
            if normalize:
                levels_copy = normalize_data(levels_copy)

            # Plot data
            if self.graph_curves is not None:
                for channel_n in range(len(levels_copy)):
                    if channel_n < len(self.graph_curves):
                        self.graph_curves[channel_n].setData(x=frequencies, y=levels_copy[channel_n])

        # Set axes range
        self.plot_set_axes_range(frequencies)

    def plot_clear(self):
        """
        Clears data
        :return:
        """
        if self.graph_curves is not None and len(self.graph_curves) > 0:
            for graph_curve in self.graph_curves:
                self.graph_widget.removeItem(graph_curve)
