
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
    #"E": 1500,  # Ethanol
    #"N": 2010,  # Natural Gas
}

# Mapping-Anzeige: Fuel Type
fuel_display_map = {
    "Regular (Benzin)": "X",
    "Premium (Benzin)": "Z",
    "Diesel": "D",
    #"Ethanol (E85)": "E",
    #"Erdgas": "N"
}
fuel_options = list(fuel_display_map.keys())

# Verhältnis Stadt/Autobahn Verbrauch als Dropdown-Optionen
verbrauchsverhaeltnis_options = {
    "Nur Autobahn (0.82)": 0.82,
    "Gemischt (1.0)": 1.0,
    "Nur Stadt (1.14)": 1.14
}

def vorhersage_co2(motor_l, zylinder, getriebeart, ganganzahl, verbrauchsverhaeltnis_label, kraftstoff_name):
    # Label in Wert umwandeln
    kraftstoff = fuel_display_map[kraftstoff_name]
    
    # Eingabedaten vorbereiten (ohne consumption_ratio!)
    input_dict = {
        'Engine Size(L)': motor_l,
        'Cylinders': zylinder,
        'is_automatic': int(getriebeart == "Automatik"),
        'is_manual': int(getriebeart == "Manuell"),
        'gear_count': ganganzahl,
    }

    # Alle Segmentspalten auf 0 setzen
    for col in feature_list:
        if col.startswith('Vehicle Segment_'):
            input_dict[col] = 0
        elif col.startswith('Fuel Type_'):
            input_dict[col] = int(col == f"Fuel Type_{kraftstoff}")

    # DataFrame erstellen und fehlende Spalten auffüllen
    input_df = pd.DataFrame([input_dict])
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = 0

    # Verbrauch gemischt vorhersagen
    verbrauch_mixed = model.predict(input_df[feature_list])[0]

    # Verhältnis-Gewichtungen Stadt/Autobahn (für nachträgliche Korrektur)
    gewichte = {
        "Nur Autobahn (0.82)": (0.3, 0.7),
        "Gemischt (1.0)":       (0.55, 0.45),
        "Nur Stadt (1.14)":     (0.8, 0.2)
    }
    gewicht_stadt, gewicht_autobahn = gewichte[verbrauchsverhaeltnis_label]

    # Stadt- und Autobahnverbrauch rekonstruieren aus gemischtem Verbrauch
    verbrauch_stadt = verbrauch_mixed * 1.14
    verbrauch_autobahn = verbrauch_mixed * 0.82

    # Endgültiger Verbrauch je nach Nutzerwahl
    verbrauch_final = verbrauch_stadt * gewicht_stadt + verbrauch_autobahn * gewicht_autobahn

    # CO2 berechnen
    co2_pred = verbrauch_final * emission_factors.get(kraftstoff, 2330) / 100

    return f"Vorhergesagter Verbrauch: {verbrauch_final:.2f} L/100 km\nGeschätzter CO₂-Ausstoss: {co2_pred:.0f} g/km"


# Gradio Interface ohne Fahrzeugsegment
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
        gr.Radio(choices=fuel_options, label="Treibstoffart")
    ],
    outputs=gr.Textbox(label="Vorhersage"),
    title="CO₂-Emissionsrechner basierend auf Verbrauch",
    description="Gib technische Fahrzeuginformationen ein, um den Kraftstoffverbrauch sowie den geschätzten CO₂-Ausstoss (g/km) vorherzusagen."
)

app.launch()
