# Sniff AI — AI-Powered Molecular Odor Detection

Sniff AI is a research prototype that predicts odor descriptors from molecular structures and visualizes the predictions through an **eight-channel simulated electronic nose**.

The project combines cheminformatics, classical machine learning, and an interactive Gradio interface. Users can enter a molecule's SMILES string or choose one of six example aroma-associated molecules and inspect predicted odor descriptors and virtual sensor responses.

> **Scope note:** The current application predicts odors from molecular structures. Its eight sensor channels are simulated from model scores; the project does not yet read physical gas sensors or identify the chemical composition of real-world smell mixtures.

## Demo

The interactive prototype was developed in Google Colab and launched with Gradio. A Colab-generated public link is temporary and only works while its runtime is active. A persistent hosted demo can be deployed with Hugging Face Spaces after the application and trained model artifact are added to this repository.

## Features

- SMILES input with RDKit validation.
- 2,048-bit Morgan fingerprints (radius 2) as molecular features.
- Random Forest multi-output odor prediction for ten target descriptors.
- Ranked odor output with model confidence scores.
- Eight simulated sensor channels visualized with Plotly.
- Six selectable example molecules: Vanilla, Coffee, Mint, Lemon, Banana, and Wood Smoke.

## Architecture

```text
Molecule SMILES / example selection
              |
              v
     RDKit molecule parsing
              |
              v
  Morgan fingerprint (2048 bits)
              |
              v
  Trained Random Forest model
              |
              +----------------------+
              |                      |
              v                      v
  Ten odor descriptor scores   Eight virtual channels
              |                      |
              +----------+-----------+
                         |
                         v
                Gradio + Plotly UI
```

### Current target descriptors

`coffee`, `vanilla`, `cooling`, `mint`, `lemon`, `citrus`, `banana`, `smoky`, `roasted`, and `woody`.

### Virtual sensor channels

The current prototype maps groups of predicted odor scores into eight display channels: Fruity, Sweet, Floral, Green, Woody, Spicy, Fresh, and Sulfur. These are illustrative software channels, not measured sensor responses or calibrated gas concentrations.

## Dataset

The development notebook used the curated odor dataset from the OpenPOM research ecosystem:

- 4,983 molecular entries were processed.
- 138 binary odor descriptor columns were identified.
- 24,312 positive molecule–odor annotations were present in the processed label matrix.
- SMILES were read from the `nonStereoSMILES` field.

Dataset source: [curated_GS_LF_merged_4983.csv](https://github.com/ARY2260/openpom/blob/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv)

Research implementation: [BioMachineLearning/openpom](https://github.com/BioMachineLearning/openpom)

The project uses a lightweight Morgan-fingerprint + Random Forest baseline. It is **not** a reproduction of OpenPOM's graph neural network architecture or its published benchmark.

## Model development and evaluation

An initial model was trained on the 20 most frequent odor labels. Thresholds were then tuned per label using a validation split.

| Evaluation | Micro F1 | Macro F1 |
|---|---:|---:|
| Initial 20-label baseline | 0.3356 | 0.2007 |
| Threshold-tuned 20-label model | 0.5368 | 0.4816 |

A separate targeted model was trained on the ten descriptors listed above and is used by the interactive application. Its final held-out evaluation metrics were **not recorded in the notebook output available for this release**, so no accuracy or F1 claim is made for that targeted model.

Confidence values shown in the UI are model scores. They are not experimentally calibrated probabilities and should not be interpreted as certainty that a real sample has a particular smell.

## Example molecules

The dashboard currently provides these representative single-molecule SMILES examples:

| UI category | Example SMILES | Note |
|---|---|---|
| Vanilla | `COc1cc(C=O)ccc1O` | Vanillin |
| Mint | `CC(C)C1CCC(C)CC1O` | Menthol-like example |
| Lemon | `CC1=CCC(CC1)C(=C)C` | Limonene |
| Coffee | `SCc1ccco1` | Representative sulfur-containing molecule |
| Banana | `CC(=O)OCCC(C)C` | Representative ester |
| Wood Smoke | `COc1ccccc1O` | Representative aromatic molecule |

These are demonstration molecules, not complete chemical mixtures extracted from coffee, mint leaves, lemon, banana, wood smoke, candles, or other real-world samples. The category labels are illustrative and should not be treated as laboratory-validated sample identities.

## Technology stack

- **Python** — application and model workflow
- **RDKit** — molecular parsing and Morgan fingerprints
- **scikit-learn** — Random Forest model
- **NumPy / Pandas** — numerical and tabular processing
- **Gradio** — interactive web UI
- **Plotly** — sensor-response visualization
- **Google Colab** — development environment

## Repository layout

The intended deployable layout is:

```text
Sniff_AI/
├── app.py                 # Gradio application
├── target_model.pkl       # Trained targeted model artifact
├── requirements.txt       # Python dependencies
├── README.md
└── assets/
    └── architecture.png   # Optional architecture illustration
```

The current repository documentation and dependency manifest are committed first. The model artifact and application source still need to be added from the working Colab notebook before this repository is independently runnable.

## Run locally

Python 3.10 or 3.11 is a practical starting point for the RDKit/scikit-learn stack. Create a virtual environment, then install:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

After `app.py` and `target_model.pkl` have been added, launch:

```bash
python app.py
```

The app should call `demo.launch()` (or `demo.launch(share=True)` for a temporary public Gradio share link). Do not commit credentials or API tokens.

## Future work

1. Add reproducible training and evaluation scripts.
2. Record held-out metrics for the ten-label targeted model.
3. Add class-wise precision, recall, F1, PR curves, and calibration analysis.
4. Integrate a physical sensor array, such as a BME688-based setup, with an ESP32 data-collection node.
5. Collect labelled sensor readings under controlled conditions (including temperature, humidity, baseline and repeat measurements).
6. Train and validate a separate sensor-input classifier. The molecular-SMILES model cannot directly consume raw gas-sensor readings without a distinct sensor-data training pipeline.
7. Compare molecular predictions with real sensor observations and laboratory references.
8. Deploy a persistent public demo using a hosting service such as Hugging Face Spaces.

## Hardware roadmap: real electronic nose

A future hardware version could use:

```text
Target sample / headspace
          |
          v
Gas / VOC sensor array
          |
          v
ESP32 (sampling + transmission)
          |
     USB / Wi-Fi
          |
          v
Laptop or edge computer
          |
          v
Sensor-data preprocessing
          |
          v
Sensor-trained ML classifier
          |
          v
Predicted odor category + dashboard
```

The laptop is the **host machine** running the application and inference. The target is the **sample being measured** (for example, coffee headspace), not another computer. The ESP32 and sensors form the sensor node. A BME688 or MQ-series sensors can provide gas/VOC-sensitive measurements, but they do not uniquely identify every odor on their own. A sensor-based model requires representative labelled readings, calibration, environmental compensation, and validation.

## Responsible use and limitations

- This is an educational/research prototype, not a certified gas detector, safety alarm, medical device, or food-quality instrument.
- Do not use it to detect toxic, flammable, or otherwise hazardous gases.
- Real-world odor perception depends on mixtures, concentration, temperature, humidity, airflow, and human perception; single-molecule predictions do not fully model these effects.
- Simulated sensor values are derived from model scores and are not physical measurements.

## Acknowledgements

- [OpenPOM](https://github.com/BioMachineLearning/openpom) and the Principal Odor Map research ecosystem for the curated odor-label dataset and research context.
- [RDKit](https://www.rdkit.org/) for cheminformatics tools.

## License

This repository does not yet include a license. Add a license before others reuse, modify, or distribute the code.
