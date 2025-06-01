# CO₂-Emissionsvorhersage mit Machine Learning

## Projektübersicht
Ziel dieses Machine-Learning-Projekts ist es, den kombinierten Kraftstoffverbrauch (in L/100 km) von Fahrzeugen basierend auf technischen Merkmalen wie Motorleistung, Antrieb, Getriebe und Fahrzeugsegment vorherzusagen. Anschliessend wird daraus der CO₂-Ausstoss berechnet. Ziel ist es, den Prozess einer End-to-End ML-Anwendung beispielhaft umzusetzen, von der Datenaufbereitung über das Feature Engineering, verschiedene Modellierungs-Iterationen, bis hin zur fertigen App.
Ursprünglich wurde der CO₂-Ausstoss direkt vorhergesagt. Aufgrund der stark abhängigen Berechnungsformel wurde später der Verbrauch als Zielgrösse verwendet, um bessere Modellierung und Vergleichbarkeit zu ermöglichen.

## Ergebnisse

Iteration 11 mit reduziertem, praxistauglichem Feature-Set wurde als finales Modell für die App gewählt.  
Das finale Random-Forest-Modell erzielt einen RMSE von **0.77 L/100 km** sowie einen R²-Wert von **0.927**.  
Trotz etwas schwächerer Metriken im Vergleich zur besten Iteration (Nr. 10) zeigt das Modell eine gute Generalisierung und keine Overfitting-Tendenzen.  
Alle gewählten Features sind realistisch abfragbar und decken die häufigsten Fahrzeug- und Antriebstypen ab.  
Für die CO₂-Schätzung wird der vorhergesagte Verbrauch mit einem kraftstoffspezifischen Emissionsfaktor multipliziert.

> 🟢 Hinweis: Das finale Modell wurde auf Robustheit und Praxistauglichkeit optimiert. Eine zukünftige Erweiterung um Fahrzeugsegmente oder Umweltdaten ist denkbar.



### Name & URL
| Name          | URL |
|--------------|----|
| Huggingface  | [Huggingface Space](https://huggingface.co/spaces/bloecand/co2-prediction-app/tree/main) |
| Code         | [GitHub Repository](https://github.com/andrinbl/co2-prediction/tree/main) |


## Datenquellen
Das Projekt basiert auf einer Kombination zweier öffentlich zugänglicher Datensätze:

| Data Source | Features |
|-------------|----------|
| [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) | Enthält technische Angaben zu Fahrzeugen, einschliesslich Motorgrösse, Getriebe, Zylinderzahl, Verbrauch und Kraftstofftyp. |
| [Wikipedia - Fahrzeugklassen](https://de.wikipedia.org/wiki/Fahrzeugklasse) | Enthält eine manuell gepflegte Klassifikation von Fahrzeugsegmenten (z. B. SUV, Kompaktklasse), basierend auf Modellnamen. |

Die Datensätze wurden im Notebook `01_eda.ipynb` bereinigt, zusammengeführt und verarbeitet. Dabei wurden u. a. fehlende Werte entfernt, das Getriebe binär kodiert und die Fahrzeugklassen mithilfe eines externen Mappings in **Fahrzeugsegmente** überführt (z. B. „SUV gross“, „Kompaktklasse“).

Die Zielvariable ist **combined_consumption (L/100 km)**, ein bereinigter Verbrauchswert. Daraus wird nach der Modellvorhersage in der App der **CO₂-Ausstoss in g/km** abgeleitet.

## Legende Datensatz
Regular Benzin - X, Premium Benzin - Z, Diesel - D, Ethanol - E, Erdgas - N


## Datenaufbereitung
- Einlesen der beiden Datensätze: technischer Fahrzeugdatensatz (Kaggle) und manuell gepflegte Fahrzeugsegmentierung (Wikipedia)
- Entfernen irrelevanter Spalten und Behandeln fehlender Werte (fillna(0))
- Erstellung eines neuen numerischen Merkmals consumption_ratio (Faktor Stadt/Autobahn-Verbrauch)
- Aufteilung in Trainings- und Testdaten (80/20), optional mit Cross-Validation zur Modellbewertung


## Feature Engineering
- Extraktion von numerischer Anzahl Gänge aus dem Getriebe-String
- Erstellung von binären Features für Automatik/Manuell
- One-Hot-Encoding für Fahrzeugsegment und Kraftstoffart
- Erstellung des neuen Features consumption_ratio basierend auf gewähltem Fahrprofil --> wurde anschliessend auskommentiert
- Finale Featureliste wird in X_columns.csv gespeichert, um später beim App-Input konsistent verwendet zu werden


## Erstellte Features

| Feature | Beschreibung |
|---------|--------------|
| `is_automatic`, `is_manual` | Binäre Kodierung der Getriebeart basierend auf Textfeldern |
| `gear_count` | Anzahl Gänge (numerisch übernommen) |
| `consumption_ratio` | Multiplikativer Faktor für das Stadt-/Autobahn-Verhältnis (basierend auf Dropdown-Auswahl in der App) --> wurde aber dann wieder aukommentiert |
| `Vehicle Segment_*` | One-Hot-Encoding der manuell zugewiesenen Fahrzeugsegmente (z. B. „SUV“, „Kompaktklasse“) |
| `Fuel Type_*` | One-Hot-Encoding der Kraftstofftypen („X“, „Z“, „D“, etc.), basierend auf Mapping-Logik |
| `Engine Size (L)` | Literangabe der Motorgrösse (numerisch übernommen) |
| `Cylinders` | Anzahl Zylinder (numerisch übernommen) |

## Model Training
### Datenmenge
- Insgesamt 6274 Fahrzeuge nach Datenbereinigung

### Methode der Datenaufteilung (Train/Test)
- Die aufbereiteten Daten wurden im Verhältnis 80 % Training und 20 % Test aufgeteilt (`train_test_split`)
- Zusätzlich kam **5-fache Cross-Validation** zur Bewertung verschiedener Modelle zum Einsatz


## Modell-Iterationen & Performance

| It. Nr | Modell         | Performance                        | Features                                                                                             | Beschreibung                                                                                                           |
|--------|----------------|-------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 1      | Linear Regression | RMSE: 20.53 g/km, R²: 0.877       | `Engine Size (L)`, `Cylinders`, `Fuel Consumption Comb`, `is_automatic`                             | Erste Version mit direkter **CO₂-Vorhersage** (alte Zielvariable)                                                     |
| 2      | Random Forest     | RMSE: 10.11 g/km, R²: 0.970       | Gleich wie Iteration 1                                                                              | Gute Performance, jedoch fraglich wegen direkter CO₂-Zielvariable                                                     |
| 3      | Linear Regression | RMSE: 1.67 L/100 km, R²: 0.670    | `Engine Size (L)`, `Cylinders`, `is_automatic`                                                      | Wechsel zur neuen Zielvariable: **Verbrauch** (L/100 km)                                                              |
| 4      | Random Forest     | RMSE: 1.51 L/100 km, R²: 0.730    | Gleich wie Iteration 3                                                                              | Gleiche Features wie vorher, **besserer Fit** als bei Linear Regression                                               |
| 5      | Linear Regression | RMSE: 1.84 L/100 km, R²: 0.600    | `Cylinders`, `is_automatic`                                                                         | `Engine Size (L)` entfernt → **deutlicher Abfall** der Modellgüte                                                     |
| 6      | Random Forest     | RMSE: 1.78 L/100 km, R²: 0.627    | Gleich wie Iteration 5                                                                              | Ebenfalls schwächer als vorher, **Feature sehr einseitig**                                                            |
| 7      | Random Forest     | RMSE: 0.49 L/100 km, R²: 0.971    | `Engine Size (L)`, `Cylinders`, `consumption_ratio`, `gear_count`, `is_automatic`, `Fuel Type_*`, `Vehicle Segment_*` | Finale Version noch ohne Cross Validation                                                                             |
| 8      | Random Forest     | RMSE: 0.62 L/100 km, R²: 0.951 (CV) | `Engine Size (L)`, `Cylinders`, `consumption_ratio`, `gear_count`, `is_automatic`, `Fuel Type_*`, `Vehicle Segment_*` | 5-fache Cross-Validation integriert, deutlicher Rückgang des RMSE und sehr stabiles Modell (kein Overfitting sichtbar) |
| 9      | Random Forest     | RMSE: 0.85 L/100 km, R²: 0.908 (CV) | `Engine Size (L)`, `consumption_ratio`, `gear_count`, `Fuel Type_E`                                 | Reduziertes Feature-Set – weniger komplex, aber Schwächen bei der Vorhersage spezieller Fahrzeugsegmente. Nicht weiterverwendet. |
| 10     | Random Forest     | RMSE: 0.66 L/100 km, R²: 0.947 (CV) | `Engine Size (L)`, `Cylinders`, `consumption_ratio`, `gear_count`, `is_automatic`, `Fuel Type_*`, `Vehicle Segment_*` | Erweiterte Version mit vollständigem Feature-Set. Sehr gute Generalisierung auch für SUV & Kleinwagen, keine Overfitting-Anzeichen sichtbar. |
| 11     | Random Forest  | RMSE: 0.77 L/100 km, R²: 0.927  <br>Ø RMSE (CV): 0.86, Ø R² (CV): 0.91 | `Engine Size (L)`, `Cylinders`, `gear_count`, `is_automatic`, `Fuel Type_X`, `Fuel Type_Z`, `Fuel Type_D`, `Vehicle Segment_Kompaktklasse`, `Vehicle Segment_Mittelklasse`, `Vehicle Segment_SUV kompakt`, `Vehicle Segment_SUV gross`, `Vehicle Segment_Kleinwagen`, `Vehicle Segment_Oberklasse` | Aktuelles Modell mit reduziertem, praxistauglichem Feature-Set. Sehr gute Generalisierung (CV), robust gegenüber Overfitting. Alle Features sind über die App sinnvoll abfragbar. |



## Modellinterpretation
- Wichtigste Features laut Feature Importance: Motogrösse, Kraftstoffart, Gänge und Zylinder
- Segment und Getriebetyp tragen zur Verbesserung bei, aber weniger dominant  
- Feature Importance wird visuell im Notebook gezeigt


## Anwendung / Deployment

- Die finale Anwendung wurde mit Gradio als Web-App umgesetzt
- Eingabefelder: Motorgrösse, Zylinder, Gänge, Getriebe, Treibstoff, Verbrauchsprofil
- Ausgabe: Vorhersage Verbrauch (L/100km) und daraus abgeleiteter CO₂-Ausstoss (g/km)


## Beispiel-Eingaben und Ergebnisse

| Testfall | Beschreibung                                 | Eingaben                                                                                               | Vorhergesagter Verbrauch | Geschätzter CO₂-Ausstoss |
|----------|----------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------|--------------------------|
| 1        | Kompaktklasse, 1.6 L, manuell, Diesel         | `1.6 L`, `4 Zylinder`, `manuell`, `6 Gänge`, `Gemischt`, `Diesel`                     | 6.49 L/100 km             | 172 g/km                 |
| 2        | SUV gross, 3.5 L, Automatik, Regular           | `3.5 L`, `6 Zylinder`, `automatik`, `8 Gänge`, `Nur Stadt`,`Regular (Benzin)`             | 12.02 L/100 km             | 280 g/km                 |
| 3        | Kleinwagen, 1.0 L, manuell, Erdgas            | `1.0 L`, `3 Zylinder`, `manuell`, `5 Gänge`, `Nur Autobahn`,`Premium (Benzin)`                    | 5.89 L/100 km             | 138 g/km                 |
| 4        | Sportlicher Kleinwagen, 5.0 L, manuell, Ethanol | `5.0 L`, `8 Zylinder`, `manuell`, `6 Gänge`, `Nur Autobahn`,`Regular (Benzin)`            | 12.96 L/100 km            | 302 g/km                 |


### Hinweise

- Die Werte stammen aus der jetzigen Version (Iteration 11) mit reduziertem Feature-Set.
- Die Ergebnisse sind plausibel im Vergleich zu realen Fahrzeugtypen und stärken die Modellvalidität.

## Lessons Learned & Reflexion

- Die Zielvariable CO₂-Ausstoss ist nicht ideal, da sie direkt aus dem Verbrauch berechnet wird. Wechsel auf Verbrauch war entscheidend
- Erst durch gezielte Erweiterungen um Einflussfaktoren wie gear_count, Fuel Type und Vehicle Segment konnte das Modell signifikant verbessert werden. Das Feature consumption_ratio wurde entfernt, da es durch explizite Verhältniswahl in der App ersetzt wurde.
- Die manuelle Validierung mit typischen Profilen (z. B. SUV, Kleinwagen) half, Schwächen in zu stark vereinfachten Modellen sichtbar zu machen.
- Das Projekt zeigt, dass bereits mit wenigen technischen Fahrzeugdaten und einem einfachen Nutzerprofil (Fahrverhalten) realistische Verbrauchs- und CO₂-Schätzungen möglich sind

## Verbesserungspotenzial

Obwohl das finale Modell bereits eine sehr hohe Genauigkeit erreicht, gibt es weitere Möglichkeiten zur Optimierung:

- Durch den Einsatz von `GridSearchCV` oder `RandomizedSearchCV` könnten die Modellparameter systematisch optimiert und dadurch sowohl Genauigkeit als auch Robustheit weiter verbessert werden.
- Zusätzliche oder aktuellere Daten – z. B. für Plug-in-Hybride, Elektrofahrzeuge oder europäische Modelle würden die Generalisierbarkeit weiter erhöhen.
- Merkmale wie Fahrzeuggewicht, Baujahr oder Luftwiderstand könnten helfen, Verbrauch noch präziser vorherzusagen.

Diese Punkte könnten in einer zukünftigen Projektiteration umgesetzt werden, um das Modell weiter zu verbessern und noch näher an reale Verbrauchswerte heranzukommen.


##  Erstellt von

- Andrin Blöchlinger (bloecand)
- Modul: AI Applications  
- Dozent: Prof. Benjamin Kühni  
- Frühjahrssemester 2025  

