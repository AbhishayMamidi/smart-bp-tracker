# BP Image Digitize

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20654470.svg)](https://doi.org/10.5281/zenodo.20654470)

This repository accompanies the paper:

"An Open Large-Scale, Real-World Dataset of Blood Pressure Device Images and a Benchmark Algorithm for Edge Transcription"

SMART-BP (Seven-segment Medical Automated Recognition and Transcription for Blood Pressure) is an edge-deployable framework for automated blood pressure image transcription. The repository also includes SMART-BP+, a model variant trained using both synthetic and real-world BP images.

The repository contains:

- SMART-BP: an edge-deployable blood pressure image transcription framework based on YOLOv8s.
- SMART-BP+: a variant trained using a combination of synthetic and real-world BP images.
- Code for blood pressure image transcription using SMART-BP and Gemini 3.0 Flash.
- A synthetic blood pressure image generation framework with automatic YOLO-format annotation generation.
- Fine-tuned model weights for both SMART-BP and SMART-BP+.
  

## Repository Structure

```text
BP_image_digitize/
│
├── Paper/
│   └── Submitted manuscript (PDF)
│
├── Saved_Models/
│   ├── SMART-BP weights
│   └── SMART-BP+ weights
│
├── Codes/
│   ├── Image_Transcription/
│   │   ├── SMART-BP/SMART-BP+ transcription pipeline
│   │   └── Gemini-based transcription pipeline
│   │
│   └── Synthetic_Image_Generation/
│       ├── Synthetic BP image generation
│       ├── Automatic YOLO annotation generation
│       └── Helper functions and utilities
│
├── App/
│   ├── Source_Code
│   └── BP_Image_Transcription.apk
│
├── LICENSE
└── README.md
```

## Citation

If you use this dataset, software, model weights, or code in your research, please cite the accompanying manuscript:

Nikookar et al. **An Open Large-Scale, Real-World Dataset of Blood Pressure Device Images and a Benchmark Algorithm for Edge Transcription.** Submitted to *Machine Learning: Health*, 2026.

A PDF of the submitted manuscript is included in the `Paper/` directory and will be updated with the final published version once available.

## License

This repository is released under the GNU Affero General Public License v3.0 (AGPL-3.0).

See the LICENSE file for details.

## Commercial Licensing

This repository contains software and model weights derived from or trained using the Ultralytics YOLOv8 framework.

Commercial users should be aware that use of this repository may require licenses from both:

1. Ultralytics (YOLOv8)
2. The authors of this repository

Ultralytics licensing information is available at: https://www.ultralytics.com/license

For commercial licensing of this repository, dataset, model weights, or related intellectual property, please contact: gari@gatech.edu

## Contact

Questions, bug reports, collaboration inquiries, and licensing requests should be directed to: gari@gatech.edu
