import pandas as pd
import numpy as np
import gradio as gr
import joblib
import os

# Modell und Featureliste laden
model_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'model.pkl')
model = joblib.load(os.path.abspath(model_path))
feature_list = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'X_columns.csv'))["columns"].tolist()

# Emissionsfaktoren in g CO2 pro Liter Kraftstoff
emission_factors = {
    "X": 2330,  # Regular gasoline
    "Z": 2350,  # Premium gasoline (angenommen leicht höher)
    "D": 2650,  # Diesel
    "E": 1500,  # Ethanol
    "N": 2010,  # Natural Gas
}

# Mapping-Anzeige: Fuel Type
fuel_display_map = {
    "Regular (Benzin)": "X",
    "Premium (Benzin)": "Z",
    "Diesel": "D",
    "Ethanol (E85)": "E",
    "Erdgas": "N"
}
fuel_options = list(fuel_display_map.keys())


# Datei vehicle_segment.csv laden
segment_map_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'vehicle_segment.csv'))

# Nur eindeutige Segmentnamen (Reihenfolge bleibt erhalten)
segment_options = segment_map_df["Segment"].drop_duplicates().tolist()


# Verhältnis Stadt/Autobahn Verbrauch als Dropdown-Optionen
verbrauchsverhaeltnis_options = {
    "Nur Autobahn (0.82)": 0.82,
    "Gemischt (1.0)": 1.0,
    "Nur Stadt (1.14)": 1.14
}

def vorhersage_co2(motor_l, zylinder, getriebeart, ganganzahl, verbrauchsverhaeltnis_label, segment, kraftstoff_name):
    # Label in Wert umwandeln
    verbrauchsverhaeltnis = verbrauchsverhaeltnis_options[verbrauchsverhaeltnis_label]
    kraftstoff = fuel_display_map[kraftstoff_name]

    # Eingabedaten vorbereiten
    input_dict = {
        'Engine Size(L)': motor_l,
        'Cylinders': zylinder,
        'is_automatic': int(getriebeart == "Automatik"),
        'is_manual': int(getriebeart == "Manuell"),
        'gear_count': ganganzahl,
        'consumption_ratio': verbrauchsverhaeltnis,
    }

    # One-Hot-Encoding der Fahrzeugsegmente
   # One-Hot-Encoding der Fahrzeugsegmente (präziser Vergleich)
    segment_matched = False
    for col in feature_list:
        if col.startswith('Vehicle Segment_'):
            if col == f"Vehicle Segment_{segment}":
                input_dict[col] = 1
                segment_matched = True
            else:
                input_dict[col] = 0
        elif col.startswith('Fuel Type_'):
            input_dict[col] = int(col == f"Fuel Type_{kraftstoff}")

# Warnung, wenn kein gültiges Segment gesetzt wurde
    if not segment_matched:
        return "Unbekanntes Fahrzeugsegment – bitte eine passende Auswahl treffen."

    # DataFrame erstellen und fehlende Spalten füllen - Stellt sicher, dass keine Spalte fehlt
    input_df = pd.DataFrame([input_dict])
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = 0

    # Verbrauch vorhersagen
    consumption_pred = model.predict(input_df[feature_list])[0]

    # CO2 berechnen
    co2_pred = consumption_pred * emission_factors.get(kraftstoff, 2330)/ 100

    return f"Vorhergesagter Verbrauch: {consumption_pred:.2f} L/100 km\nGeschätzter CO₂-Ausstoss: {co2_pred:.0f} g/km"

# Gradio Interface
app = gr.Interface(
    fn=vorhersage_co2,
    inputs=[
        gr.Number(label="Motorgrösse (Liter)"),
        gr.Number(label="Anzahl Zylinder", minimum=2, maximum=16),
        gr.Radio(choices=["Automatik", "Manuell"], label="Getriebeart", value="Automatik"),
        gr.Number(label="Anzahl Gänge", minimum=1),
        gr.Dropdown(label="Verhältnis Stadt/Autobahn Verbrauch",
                    choices=list(verbrauchsverhaeltnis_options.keys()),
                    value="Gemischt (1.0)"),
        gr.Radio(choices=segment_options, label="Fahrzeugsegment"),
        gr.Radio(choices=fuel_options, label="Treibstoffart")
    ],
    outputs=gr.Textbox(label="Vorhersage"),
    title="CO₂-Emissionsrechner basierend auf Verbrauch",
    description="Gib technische Fahrzeuginformationen ein, um den Kraftstoffverbrauch sowie den geschätzten CO₂-Ausstoss (g/km) vorherzusagen."
)

app.launch()
