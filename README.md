# TB Cough Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
</div>

A modern, user-friendly application for detecting tuberculosis (TB) from cough audio samples using deep learning.

## Features

- **Intuitive Interface**: Modern, clean UI design for ease of use
- **Real-time Recording**: Record cough samples directly through the application
- **Audio Visualization**: See waveforms and spectrograms of recorded coughs
- **Batch Processing**: Analyze multiple audio files at once
- **Detailed Results**: Get comprehensive analysis with confidence scores
- **Export Functionality**: Save results to CSV for further analysis
- **Cross-platform**: Works on Windows, macOS, and Linux

## Overview

The TB Cough Detection System uses a state-of-the-art deep learning model to analyze cough audio recordings and predict the likelihood of tuberculosis. The system employs a MobileNetV4 architecture with Res2TSM (Residual 2D Temporal Shift Module) to process mel spectrograms of cough audio.

## Quick Start

### Prerequisites

- Python 3.7 or higher
- Audio input device (microphone)

### Installation

#### Option 1: Run from source

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/tb-cough-detection.git
   cd tb-cough-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python inference.py
   ```

#### Option 2: Use the executable (Windows)

1. Download the latest release from the [releases page](https://github.com/yop-dev/tb-cough-detection/releases)
2. Extract the ZIP file
3. Run `TB_Cough_Detection.exe`

## Usage Guide

1. **Record Coughs**:

   - Click the "Record Cough" button
   - Follow the countdown and cough when prompted
   - Record at least 5 cough samples for accurate analysis

2. **Load Audio Files**:

   - Click "Select Audio Files" to load existing cough recordings
   - Supported formats: WAV, MP3, FLAC, OGG

3. **Process Files**:

   - After recording or loading at least 5 cough samples, click "Process Files"
   - The system will analyze each sample and provide results

4. **View Results**:

   - Results will be displayed in the main window
   - Detailed information for each sample appears in the table below
   - The overall diagnosis is determined by majority vote

5. **Export Results**:
   - Click "Export Results" to save the analysis as a CSV file
   - The export includes individual results and a summary

## Building the Executable

To build the executable yourself:

1. Install PyInstaller:

   ```bash
   pip install pyinstaller
   ```

2. Run the build script:

   ```bash
   python build_exe.py
   ```

3. The executable will be created in the `dist` folder

## Model Information

The system uses a MobileNetV4 backbone with a Res2TSM block for temporal modeling. The model processes mel spectrograms of cough audio and has been trained on a dataset of TB-positive and TB-negative cough samples.

Key components:

- **MobileNetV4**: Efficient CNN architecture for feature extraction
- **Res2TSM**: Temporal modeling for capturing time-dependent features
- **Mel Spectrogram**: Audio representation that mimics human hearing perception

## Performance

The model achieves:

- Sensitivity: ~85%
- Specificity: ~90%
- Accuracy: ~88%

_Note: Performance metrics are based on internal testing and may vary in real-world scenarios._

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MobileNetV4 implementation is based on the [TIMM library](https://github.com/rwightman/pytorch-image-models)
- The Res2TSM module is inspired by the Temporal Shift Module and Res2Net architectures
- Thanks to all contributors who have helped improve this project

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on this repository.

---

<div align="center">
  <sub>Built with love for healthcare innovation</sub>
</div>
