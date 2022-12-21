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
import pyqtgraph
from pyqtgraph import mkPen

from AudioHandler import normalize_data, apply_reference

GRAPH_DBFS_RANGE_FROM = -60
GRAPH_DBFS_RANGE_TO = 10
GRAPH_DBFS_NORM_RANGE_FROM = -40
GRAPH_DBFS_NORM_RANGE_TO = 30
GRAPH_DISTORTIONS_RANGE_FROM = -80
GRAPH_DISTORTIONS_RANGE_TO = -10


class GraphPlot:
    def __init__(self, graph_widget, settings_handler):
        """
        Initializes GraphPlot class
        :param graph_widget: PyQt pyqtgraph GraphWidget
        :param settings_handler: SettingsHandler class
        """
        self.graph_widget = graph_widget
        self.settings_handler = settings_handler

        self.frequency_response_plotter = None
        self.distortions_plotter = pyqtgraph.ViewBox()

        # Plots
        self.graph_curves_frequency_responses = []
        self.graph_curves_distortions = []

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

        self.frequency_response_plotter = self.graph_widget.plotItem
        self.frequency_response_plotter.scene().addItem(self.distortions_plotter)
        self.distortions_plotter_update_view()
        self.frequency_response_plotter.getAxis('right').linkToView(self.distortions_plotter)
        self.distortions_plotter.setXLink(self.frequency_response_plotter)
        self.frequency_response_plotter.vb.sigResized.connect(self.distortions_plotter_update_view)

        # Set axes name
        self.graph_widget.setLabel('left', 'Level', units='dBFS')
        self.graph_widget.setLabel('right', 'Total harmonic distortion, IEEE', units='dB')
        self.graph_widget.setLabel('bottom', 'Frequency', units='Hz')

        # Set axes range
        self.plot_set_axes_range(frequency_list)

        # Enable log mode on frequency scale
        self.frequency_response_plotter.getAxis('bottom').setLogMode(True)
        self.frequency_response_plotter.getAxis('left').setLogMode(False)
        self.frequency_response_plotter.getAxis('right').setLogMode(False)

    def distortions_plotter_update_view(self):
        self.distortions_plotter.setGeometry(self.frequency_response_plotter.vb.sceneBoundingRect())

    def plot_set_axes_range(self, frequency_list, normalize=False):
        """
        Set axes range of plot widget
        :param frequency_list: list of frequencies from AudioHandler class
        :param normalize:
        :return:
        """
        # Calculate frequency range
        if frequency_list is not None and len(frequency_list) > 0:
            min_frequency = max(frequency_list[0], 1)
            max_frequency = max(frequency_list[-1], 1)
        else:
            min_frequency = 1
            max_frequency = int(self.settings_handler.settings['audio_sample_rate']) // 2

        # Set ranges
        self.frequency_response_plotter.setXRange(math.log10(min_frequency), math.log10(max_frequency), padding=0)
        if normalize:
            self.frequency_response_plotter.setYRange(GRAPH_DBFS_NORM_RANGE_FROM, GRAPH_DBFS_NORM_RANGE_TO)
        else:
            self.frequency_response_plotter.setYRange(GRAPH_DBFS_RANGE_FROM, GRAPH_DBFS_RANGE_TO)
        self.distortions_plotter.setYRange(GRAPH_DISTORTIONS_RANGE_FROM, GRAPH_DISTORTIONS_RANGE_TO)

    def plots_prepare(self, recording_channels):
        """
        Creates new plots
        :param recording_channels:
        :return:
        """
        # Remove previous plots
        self.plot_clear()

        # Add new plots
        self.graph_curves_frequency_responses = []
        self.graph_curves_distortions = []
        for channel_n in range(recording_channels):
            plot_color_level \
                = int(255 - (channel_n / recording_channels) * 255), int((channel_n / recording_channels) * 255), 0
            plot_color_distortion \
                = int(127 - (channel_n / recording_channels) * 127), int((channel_n / recording_channels) * 127), 127
            # Levels
            self.graph_curves_frequency_responses.append(
                self.frequency_response_plotter.plot(pen=mkPen(color=plot_color_level),
                                                     name='Channel ' + str(channel_n + 1)))

            # Distortions
            distortions_item = pyqtgraph.PlotCurveItem(pen=mkPen(color=plot_color_distortion))
            self.distortions_plotter.addItem(distortions_item)
            self.graph_curves_distortions.append(distortions_item)
        if recording_channels > 1:
            # Levels sum
            self.graph_curves_frequency_responses.append(
                self.frequency_response_plotter.plot(pen='k', name='Sum'))

            # Distortions sum
            distortions_sum_item = pyqtgraph.PlotCurveItem(pen='k')
            self.distortions_plotter.addItem(distortions_sum_item)
            self.graph_curves_distortions.append(distortions_sum_item)

    def plot_data(self, frequencies, levels, distortions, normalize: bool,
                  ref_frequencies=None, ref_levels=None, ref_distortions=None, normalize_ref=False):
        """
        Plots data on graph
        :param frequencies: list of frequencies from AudioHandler class
        :param levels: list of levels from AudioHandler class
        :param distortions: list of distortions from AudioHandler class
        :param normalize: normalize data?
        :param ref_frequencies: list of frequencies from AudioHandler class
        :param ref_levels: list of levels from AudioHandler class
        :param ref_distortions: list of distortions from AudioHandler class
        :param normalize_ref: normalize reference data before applying it
        :return:
        """
        if levels is not None and len(levels) > 0:
            # Coppy arrays
            levels_copy = levels.copy()
            if len(distortions) > 0:
                distortions_copy = distortions.copy()
            else:
                distortions_copy = None

            # Apply reference?
            if ref_frequencies is not None and ref_levels is not None \
                    and len(ref_frequencies) > 0 and len(ref_levels) > 0:
                levels_copy = apply_reference(frequencies, levels_copy, ref_frequencies, ref_levels, normalize_ref)

            # Apply reference?
            if distortions_copy is not None and ref_frequencies is not None and ref_distortions is not None \
                    and len(ref_frequencies) > 0 and len(ref_distortions) > 0:
                distortions_copy = apply_reference(frequencies, distortions, ref_frequencies, ref_distortions, True)

            # Calculate channel sum if channels > 1
            if len(levels_copy) > 1:
                levels_copy = np.vstack((levels_copy, np.average(levels_copy, axis=0)))

            # Normalize?
            if normalize:
                levels_copy = normalize_data(levels_copy, frequencies)

            # Calculate distortions sum
            if distortions_copy is not None:
                distortions_copy = np.vstack((distortions_copy, np.average(distortions_copy, axis=0)))

            # Plot data
            if self.graph_curves_frequency_responses is not None:
                for channel_n in range(len(levels_copy)):
                    if channel_n < len(self.graph_curves_frequency_responses):
                        # Plot frequency response
                        self.graph_curves_frequency_responses[channel_n] \
                            .setData(x=np.log10(frequencies), y=levels_copy[channel_n])

                        # Plot distortions
                        if distortions_copy is not None:
                            self.graph_curves_distortions[channel_n] \
                                .setData(x=np.log10(frequencies), y=distortions_copy[channel_n])

        # Set axes range
        self.plot_set_axes_range(frequencies, normalize)

    def plot_clear(self):
        """
        Clears data
        :return:
        """
        if self.graph_curves_frequency_responses is not None and len(self.graph_curves_frequency_responses) > 0:
            for graph_curve in self.graph_curves_frequency_responses:
                self.frequency_response_plotter.removeItem(graph_curve)
        if self.graph_curves_distortions is not None and len(self.graph_curves_distortions) > 0:
            for distortions_item in self.graph_curves_distortions:
                self.distortions_plotter.removeItem(distortions_item)
