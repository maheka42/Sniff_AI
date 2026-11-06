from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Keep this order identical to the target model's training label order.
TARGET_LABELS = [
    "coffee",
    "vanilla",
    "cooling",
    "mint",
    "lemon",
    "citrus",
    "banana",
    "smoky",
    "roasted",
    "woody",
]

SAMPLES = {
    "Vanilla": "COc1cc(C=O)ccc1O",       # Vanillin
    "Coffee": "SCc1ccco1",               # Representative sulfur-containing molecule
    "Mint": "CC(C)C1CCC(C)CC1O",         # Menthol
    "Lemon": "CC1=CCC(CC1)C(=C)C",       # Limonene
    "Banana": "CC(=O)OCCC(C)C",          # Isoamyl acetate
    "Wood Smoke": "COc1ccccc1O",         # Guaiacol
}

SENSOR_MAPPING = {
    "Fruity": ["banana", "lemon", "citrus"],
    "Sweet": ["vanilla", "banana"],
    "Floral": ["vanilla"],
    "Green": ["mint", "lemon"],
    "Woody": ["woody"],
    "Spicy": ["smoky", "roasted"],
    "Fresh": ["cooling", "mint"],
    "Sulfur": ["coffee", "smoky"],
}

MODEL_PATH = Path(__file__).with_name("target_model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Add target_model.pkl beside app.py. "
        "Export it from the training notebook with "
        "joblib.dump(target_model, 'target_model.pkl')."
    )

target_model = joblib.load(MODEL_PATH)
fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def molecule_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert a valid SMILES string to a 2,048-bit Morgan fingerprint."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES. Check the molecular structure and try again.")

    fingerprint = fingerprint_generator.GetFingerprint(molecule)
    array = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array.reshape(1, -1)


def positive_class_probability(probability_output, classes, label_index: int) -> float:
    """Extract the probability for class 1 from a multi-output RF model."""
    class_values = list(classes[label_index])
    if 1 not in class_values:
        return 0.0

    positive_index = class_values.index(1)
    return float(probability_output[label_index][0][positive_index])


def predict_odors(smiles: str):
    features = molecule_to_fingerprint(smiles)
    probability_output = target_model.predict_proba(features)

    rows = []
    for index, label in enumerate(TARGET_LABELS):
        score = positive_class_probability(
            probability_output, target_model.classes_, index
        )
        rows.append({"Odor": label, "Confidence (%)": round(score * 100, 2)})

    odor_frame = (
        pd.DataFrame(rows)
        .sort_values("Confidence (%)", ascending=False)
        .reset_index(drop=True)
    )
    return odor_frame


def sniff_dashboard(sample_name: str, smiles: str):
    """Run molecular odor inference and draw eight illustrative virtual channels."""
    if not smiles or not smiles.strip():
        raise gr.Error("Enter a SMILES string or select an example sample.")

    try:
        odor_frame = predict_odors(smiles.strip())
    except ValueError as error:
        raise gr.Error(str(error)) from error

    confidence_by_label = dict(
        zip(odor_frame["Odor"], odor_frame["Confidence (%)"])
    )

    sensor_rows = []
    for sensor_name, odor_labels in SENSOR_MAPPING.items():
        channel_scores = [confidence_by_label[label] for label in odor_labels]
        sensor_rows.append(
            {
                "Sensor": sensor_name,
                "Response (%)": round(float(np.mean(channel_scores)), 2),
            }
        )

    sensor_frame = pd.DataFrame(sensor_rows)
    figure = px.bar(
        sensor_frame,
        x="Sensor",
        y="Response (%)",
        range_y=[0, 100],
        title="Eight-channel simulated electronic nose",
    )
    figure.update_layout(xaxis_tickangle=-25)

    top_row = odor_frame.iloc[0]
    summary = (
        f"### Analysis complete: {sample_name or 'Custom molecule'}\n"
        f"**Top predicted odor:** {top_row['Odor'].title()}  \n"
        f"**Model score:** {top_row['Confidence (%)']:.2f}%"
    )

    return summary, odor_frame, figure


def load_sample(sample_name: str) -> str:
    return SAMPLES.get(sample_name, "")


with gr.Blocks(theme=gr.themes.Soft(), title="Sniff AI") as demo:
    gr.Markdown(
        """
        # Sniff AI
        ### AI-Powered Molecular Odor Detection

        Enter a molecule's SMILES or select an example. Sniff AI predicts
        odor descriptors and displays eight **simulated** sensor channels.
        """
    )

    sample_dropdown = gr.Dropdown(
        choices=list(SAMPLES.keys()),
        value="Vanilla",
        label="Choose a smell sample",
    )
    smiles_input = gr.Textbox(
        value=SAMPLES["Vanilla"],
        label="Molecule SMILES",
        lines=2,
    )
    sample_dropdown.change(
        fn=load_sample,
        inputs=sample_dropdown,
        outputs=smiles_input,
    )

    sniff_button = gr.Button("SNIFF NOW", variant="primary")
    summary_output = gr.Markdown()
    odor_output = gr.Dataframe(
        headers=["Odor", "Confidence (%)"],
        label="Predicted odor profile",
        interactive=False,
    )
    sensor_output = gr.Plot(label="Virtual sensor reactions")

    sniff_button.click(
        fn=sniff_dashboard,
        inputs=[sample_dropdown, smiles_input],
        outputs=[summary_output, odor_output, sensor_output],
    )

    gr.Markdown(
        """
        ---
        **Technology:** RDKit | Morgan fingerprints | Random Forest | Gradio

        **Research prototype:** model scores are estimates. Virtual sensor
        channels are computed from those scores; they are not physical gas
        sensor readings or experimentally calibrated concentrations.
        """
    )


if __name__ == "__main__":
    demo.launch()
