# SonicEval (aka Pulsely)
## Audio Quality Measurement Software
***Note:** **SonicEval** was previously named **Pulsely**. I received a message from GitHub Support who thinks that this name violates some kind of copyright, and this project will be removed. Therefore, I had to rename it to **SonicEval***.

<div style="width:100%;text-align:center;">
    <p align="center">
        <a href="https://github.com/F33RNI/SonicEval/releases"><img alt="Download release" src="https://img.shields.io/badge/-Download%20latest-yellowgreen?style=for-the-badge&logo=github" ></a>
    </p>
    <p align="center">
        <a href="https://twitter.com/f33rni"><img alt="Twitter" src="https://img.shields.io/twitter/url?label=My%20twitter&style=social&url=https%3A%2F%2Ftwitter.com%2Ff33rni" ></a>
        <img src="https://badges.frapsoft.com/os/v1/open-source.png?v=103" >
        <a href="https://soundcloud.com/f3rni"><img alt="SoundCloud" src="https://img.shields.io/badge/-SoundCloud-orange" ></a>
    </p>
</div>

|![](icon.png)  |  ![](Screenshot_1.png)|
|:-------------------------:|:-------------------------:|


----------

### Description

**SonicEval** is a free program that allows you to accurately and professionally measure the frequency response, 
harmonic distortion of audio systems. Also, it can do reference measurements and calculate approximate quality score

Currently, there are 2 types of frequency response measurements:
- Using frequency sweep (maximum accuracy, but annoying sound)
- Using white noise (does not provide high accuracy, but allows you to quickly measure the frequency response without annoying sounds)

In sweep mode **SonicEval** also measures THD (total harmonic distortions) with IEEE method in %

*App and logo created by Fern Lane (aka. F3RNI)*

----------

### How to use it

1. Connect your audio interface to PC and run SonicEval app
2. In **Audio interface** section select your output and input devices
3. Specify sample rate, start and stop frequencies, test duration and other parameters
4. If possible, connect the output of your audio interface to the input. Run the test and save the result. This is required for calibration
5. Connect your audio system to the interface *(for example, the output of the audio interface connect to the speakers whose frequency response and THD you want to measure, and the input to high-quality linear microphone)*
6. Select calibration file (if available) in the **Reference** section
7. Start measurement and wait
8. Done! You can save frequency response and THD to CSV file or export as PNG image *(it is recommended to open the application in full screen for an image in a higher resolution)*

----------

### Understanding the Results

If you used a sweep (recommended) then the result should be 2 charts:
- frequency response (level in dBFS (decibels relative to full scale) *vs* frequency)
- total harmonic distortion (ratio of harmonics to fundamental in dB *vs* frequency)

Basically, you should set up your system so that the frequency response graph is as flat as possible 
and the distortion graph as low as possible

But there are nuances. For example, a frequency response graph might have a sharp rolloff at the very end. 
This is due to the built-in filters in the ADC. Also, the distortion curve may have strange peaks at the beginning 
(up to 200Hz, sometimes even higher). Often this is due to the fact that the resolution of the FFT 
does not allow you to accurately determine the levels of all harmonics and the signal fluctuates.

----------

### Run/compile from source

### Windows

- Download and install Python and pip https://www.python.org/downloads/
- Download and unpack source code from https://github.com/F33RNI/SonicEval
- Open terminal in `SonicEval-master` folder and run `pip install -r requirements.txt`
- To run SonicEval: `python SonicEval.py`


- To build, firstly update/install pyinstaller to the **latest** version: `pip install pyinstaller --upgrade`
  - you can check version by typing `pyinstaller --version`. It should be >= 5.7.0
- Install msvc-runtime to fix matplotlib error: `pip install msvc-runtime`
- Run build script: `python Builder.py`
- Compiled program is located in folder `./dist/SonicEval-x.x.x-Windows_...`
  - to run it, just double-click on `SonicEval.exe` file

### Linux

- Install Python and pip: `sudo apt update && sudo apt install python3.10`
- Install portaudio: `sudo apt install portaudio19-dev python3-pyaudio`
- Clone source code: `git clone https://github.com/F33RNI/SonicEval`
- Install requirements: `pip install -r requirements.txt`
- To run SonicEval: `python SonicEval.py`


- To build, firstly update/install pyinstaller to the **latest** version: `sudo pip install pyinstaller --upgrade`
  - you can check version by typing `pyinstaller --version`. It should be >= 5.7.0
- Run build script: `python Builder.py`
  - type `y` (yes) if it asks `WARNING: The output directory "./build/SonicEval" and ALL ITS CONTENTS will be REMOVED! Continue? (y/N)`
- Compiled program is located in folder `./dist/SonicEval-x.x.x-Linux_...`
  - to run it, just double-click on `SonicEval` file

### MacOS

- Download and install python3: https://www.python.org/downloads/macos/
- Install Homebrew: https://brew.sh/
- Install portaudio with brew: `brew install portaudio`
- Clone source code: `git clone https://github.com/F33RNI/SonicEval`
- Install requirements: `pip3 install -r requirements.txt`
- To run SonicEval: `python3 SonicEval.py`


- Building an app on a MacOS 💩 is probably impossible