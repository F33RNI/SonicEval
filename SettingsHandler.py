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

import json
import os

import AudioHandler

# Default app settings
SETTINGS_DEFAULT = {
    'audio_playback_interface': '',
    'audio_recording_interface': '',
    'audio_recording_channels': 2,
    'audio_sample_rate': 44100,

    'signal_type': AudioHandler.TEST_SIGNAL_TYPE_SWEEP,
    'signal_start_freq': 10,
    'signal_stop_freq': 22000,
    'signal_test_duration': 20,
    'audio_playback_volume': 70,

    'fft_size_chunks': 4,
    'fft_window_type': AudioHandler.WINDOW_TYPE_HAMMING,

    'noise_filter_order': 5,
    'noise_random_seed': 0,

    'csv_separator': ',',

    'normalize_frequency_response': True,
    'normalize_reference': True,
    'normalize_to_save': True
}


class SettingsHandler:
    def __init__(self, filename: str):
        """
        Initializes SettingsHandler class
        :param filename:
        """
        self.filename = filename
        self.settings = {}

    def read_from_file(self):
        """
        Parses and checks settings from file
        :return:
        """
        try:
            # Create new if file not exists
            if not os.path.exists(self.filename):
                self.settings = SETTINGS_DEFAULT
                self.write_to_file()

            # Open file
            settings_file = open(self.filename, 'r')

            # Parse JSON
            try:
                self.settings = json.load(settings_file)
            except:
                self.settings = SETTINGS_DEFAULT
                self.write_to_file()

            # Check settings
            if not self.check_settings():
                self.settings = SETTINGS_DEFAULT
                self.write_to_file()

        except Exception as e:
            print(e)

    def check_settings(self):
        """
        Checks settings
        :return:
        """
        try:
            default_keys = SETTINGS_DEFAULT.keys()
            for key in default_keys:
                if key not in self.settings:
                    return False
            return True
        except Exception as e:
            print(e)
            return False

    def write_to_file(self):
        """
        Writes settings to JSON file
        :return:
        """
        try:
            # Open file for writing
            settings_file = open(self.filename, 'w')

            # Check if file is writable
            if settings_file.writable():
                # Write json to file
                json.dump(self.settings, settings_file, indent=4)
            else:
                settings_file.close()
        except Exception as e:
            print(e)
