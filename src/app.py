#!/usr/bin/env python3

# Standard library modules
from typing import Sequence

# Third-party modules
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex

import torch

import streamlit as st


# Local application modules
from config import CFG, Net
from data import data, ldata
from utils import ISO639_LANGUAGE_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ = data.train_test_split()  # just to make sure the CountVectorizer is fitted

# To make a reproducible output
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)


# Load your model
@st.cache_resource  # Cache the model loading for efficiency
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    return model


# Function to predict code-switching
def predict_code_switching(text):
    """Predict if the input text exhibits code-switching behavior."""
    pass


# Generate language colors
def generate_language_colors(labels: Sequence):
    palette = sns.color_palette("husl", len(labels))  # or "Set2", "tab10", etc.
    hex_colors = [to_hex(color) for color in palette]
    return dict(zip(labels, hex_colors))


# Map language tags to colors
LANGUAGE_COLORS = generate_language_colors(data.label_names)


# Function to generate colored HTML for tokens
def colorize_text(text, label):
    """Create a text wrapped in HTML spans with colors based on the label."""
    colored_text = ""
    if isinstance(label, str):
        color = LANGUAGE_COLORS.get(label, None)
    elif isinstance(label, int):
        label = st.session_state.le.inverse_transform([label])[0]
        color = LANGUAGE_COLORS.get(label, None)
    if color is None:
        color = "#999999"  # Default to gary if label not found
    colored_text += (
        f'<span style="background-color: {color}; padding: 3px; margin: 2px; border-radius: 2px;">{text}</span> '
    )
    return colored_text


@st.cache_resource
def load_data():
    X = ldata.X.loc[ldata.raw_y.isin(ISO639_LANGUAGE_NAMES.keys())]
    y = ldata.raw_y.loc[ldata.raw_y.isin(ISO639_LANGUAGE_NAMES.keys())]

    rx = st.session_state.cv.transform([i for i in X]).toarray()
    rx = torch.tensor(rx, dtype=torch.float).to(device)

    ry = [st.session_state.le.transform([ISO639_LANGUAGE_NAMES[i]]) for i in y]

    return pd.DataFrame({"Text": X, "Label": ry})


# Streamlit App Layout
def main():
    # Sidebar
    st.sidebar.markdown(
        "The supported languages are:<br/><br/>"
        + "".join(
            [
                f"{i:02d}. {colorize_text(text, label)}<br/>"
                for i, (text, label) in enumerate(zip(data.label_names + ["Other"], data.label_names + ["Other"]))
            ]
        ),
        unsafe_allow_html=True,
    )

    # Title and Description
    st.title("Language Detection")
    st.markdown("""
    This application uses a Character Based bi/trigram for document language prediction. 
    Enter a sentence or paragraph below to see the result.

    """)

    st.markdown("### Select a Row")
    df = load_data()
    # Add a selection column
    df["select"] = False
    selected_rows = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_order=["select", "Text", "Label"],  # + list(df.columns),
        key="data_editor",
        column_config={
            "select": st.column_config.CheckboxColumn("Select", help="Select rows to keep"),
        },
    )

    st.markdown("### OR Enter text to analyze")
    user_input = st.text_area("", height=200, key="user_input")

    no_selected_rows = False
    no_user_input = False

    # Button to Predict
    if st.button("Analyze"):
        lines = []

        # Filter selected rows
        selected_rows = selected_rows[selected_rows["select"] == True]

        if not selected_rows.empty:
            lines.extend(selected_rows["Text"].tolist())
        else:
            no_selected_rows = True

        if user_input.strip():
            with st.spinner("Analyzing..."):
                st.subheader("Results:")
                for line in user_input.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    lines.append(line)
        else:
            no_user_input = True

        if no_selected_rows and no_user_input:
            st.warning("Please enter text or select a row before clicking analyze.")

        if lines:
            df = pd.DataFrame(columns=["Text", "Predicted Label"])
            processed_lines = pd.DataFrame.from_dict({"Text": lines})
            processed_lines = data.preprocess_text(processed_lines["Text"])

            for line, pline in zip(lines, processed_lines):
                # text preprocessing
                x = st.session_state.cv.transform([pline]).toarray()
                x = torch.tensor(x, dtype=torch.float).to(device)

                # Predict
                with torch.no_grad():
                    label = st.session_state.model(x)
                    label = label.argmax(dim=1).item()
                    label = st.session_state.le.inverse_transform([label])[0]

                # Generate Colored Text
                colored_text = colorize_text(line, label)
                new_row = {"Text": colored_text, "Predicted Label": label}
                df.loc[len(df)] = new_row

            # Display Results
            html_table = df.to_html(escape=False, index=False)  # Don't escape HTML content
            st.markdown(html_table, unsafe_allow_html=True)
            lines.clear()


# Run the app
if __name__ == "__main__":
    # Ensure model path is correct
    model_path = CFG.saved_models_path / "bestmodel_nn.pth"
    if model_path.exists():
        # model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        # Restore full pipeline
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        input_size, output_size = checkpoint["model_in_out"]
        st.session_state.model = Net(input_size, output_size)
        st.session_state.model.load_state_dict(checkpoint["model_state_dict"])
        st.session_state.model.eval()

        st.session_state.cv = checkpoint["vectorizer"]
        st.session_state.le = checkpoint["label_encoder"]
        main()
    else:
        st.error(f"Model file not found at {model_path}. Please check the path and try again.")
