# TB Cough Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
</div>

A modern, user-friendly application for detecting tuberculosis (TB) from cough audio samples using deep learning.

> **IMPORTANT DISCLAIMER**: This tool is intended for screening purposes only and is not a substitute for professional medical diagnosis. The technology for cough analysis is still in its early development phases, and results should be interpreted with caution. Always consult with healthcare professionals for proper diagnosis and treatment of tuberculosis or any other medical condition.

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
   python new_inference.py
   ```

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


## Model Information

The system uses a pre-trained MobileNetV4 backbone (from the TIMM library) with our custom Res2TSM block added after the final feature map for temporal modeling. The model processes mel spectrograms of cough audio and has been trained on the CODA TB Dream Challenge dataset from Synapse, which contains TB-positive and TB-negative cough samples.

The model architecture was specifically chosen to be mobile-friendly and resource-efficient. We deliberately avoided using transformer models or other resource-intensive architectures to ensure the system could run effectively on mobile devices and computers with limited computational resources.

Key components:

- **MobileNetV4**: Standard pre-trained efficient CNN architecture for feature extraction, optimized for mobile and edge devices (we use the implementation from the TIMM library)
- **Custom Res2TSM Block**: Our lightweight temporal modeling module added after MobileNet's feature extraction, designed for capturing time-dependent features without the computational overhead of transformer models
- **Mel Spectrogram**: Audio representation that mimics human hearing perception while being computationally efficient to process

*Note: We do not claim the MobileNetV4 architecture as our own work. We have simply utilized the pre-trained model from the TIMM library and added our custom Res2TSM block to adapt it for temporal audio analysis.*

## Performance

When evaluated on the CODA TB Dream Challenge dataset from Synapse, the model achieves:

- Sensitivity: ~85%
- Specificity: ~90%
- Accuracy: ~88%

_Note: Performance metrics are based on internal testing using the CODA TB Dream Challenge dataset and may vary in real-world scenarios. These results should not be interpreted as clinical validation. This technology is still experimental and in early development stages._

## Limitations and Current State

This TB Cough Detection System is a research tool that demonstrates the potential of audio-based screening for tuberculosis. However, users should be aware of the following limitations:

- **Early Development Stage**: The technology for cough analysis is still emerging and not fully validated in diverse clinical settings.
- **Screening Tool Only**: This system is designed to assist with preliminary screening and should never replace proper medical testing and diagnosis.
- **Environmental Factors**: Background noise, recording quality, and other environmental factors can significantly affect results.
- **Limited Training Data**: The model has been trained on the CODA TB Dream Challenge dataset from Synapse, which may not represent all populations, cough types, or TB manifestations.
- **Not FDA/CE Approved**: This tool has not received regulatory approval for clinical use in diagnosing tuberculosis.

Healthcare professionals should consider this as one of many tools in the diagnostic process, and patients should always seek proper medical evaluation regardless of the results provided by this system.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MobileNetV4 implementation is directly from the [TIMM library](https://github.com/rwightman/pytorch-image-models) by Ross Wightman and contributors
- The Res2TSM module is inspired by the [Temporal Shift Module](https://github.com/mit-han-lab/temporal-shift-module) by MIT HAN Lab and the [Res2Net](https://github.com/gasvn/Res2Net) architecture by Shang-Hua Gao et al.
- The model was trained using the [CODA TB Dream Challenge dataset](https://www.synapse.org/#!Synapse:syn26133770) provided by Synapse
- We thank the original authors of these architectures and libraries, as well as Synapse for providing the dataset for research purposes
- Thanks to all contributors who have helped improve this project

## Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- **Joner De Silva**
  - LinkedIn: [https://www.linkedin.com/in/joner-de-silva-861575203/](https://www.linkedin.com/in/joner-de-silva-861575203/)
  - Portfolio: [https://portfolio-theta-two-19.vercel.app](https://portfolio-theta-two-19.vercel.app)
  
You can also open an issue on this repository for project-specific inquiries.

---

<div align="center">
  <sub>Built with love for healthcare innovation</sub>
</div>
