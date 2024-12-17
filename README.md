# **Bildbasierte Analyse und Visualisierung von Merkmalen**

Dieses Projekt bietet eine umfassende Lösung zur Analyse und Visualisierung von Bilddaten. Es integriert Module zur **Bildsegmentierung**, **Merkmalsextraktion**, **Vergleich von Merkmalen**, **3D-Visualisierung von Embeddings** sowie zur **Vergleichsanalyse neuer Daten mit bestehenden Quellen**.

---

## **Inhaltsverzeichnis**
1. [Übersicht](#übersicht)
2. [Hauptfunktionen](#hauptfunktionen)
3. [Verzeichnisstruktur](#verzeichnisstruktur)
4. [Systemanforderungen](#systemanforderungen)
5. [Installationsanleitung](#installationsanleitung)
6. [Verwendung der Module](#verwendung-der-module)
7. [Beispielablauf](#beispielablauf)
8. [Erklärung der Module](#erklärung-der-module)
9. [Screenshots und Beispiele](#screenshots-und-beispiele)
10. [Lizenz](#lizenz)
11. [Kontakt](#kontakt)

---

## **Übersicht**

Dieses Projekt wurde entwickelt, um Defekte oder Merkmale in Bildern zu analysieren und zu vergleichen. Es unterstützt mehrere Anwendungsbereiche wie:

- **Industrieproduktion:** Erkennung von Defekten in mechanischen Komponenten.
- **Agrarwissenschaft:** Analyse von Pflanzenkrankheiten oder Holzdefekten.
- **Medizin:** Identifikation von Hauterkrankungen.

Das Tool bietet eine **grafische Benutzeroberfläche (GUI)** für einfaches Arbeiten sowie mehrere Python-Skripte für erweiterte Analysen.

---

## **Hauptfunktionen**

**1. Interaktive Bildsegmentierung**
- Ermöglicht die manuelle Auswahl und Bearbeitung von Regionen in Bildern.
- Speichert segmentierte Regionen als separate Bilddateien zur Weiterverarbeitung.

**2. Merkmalsextraktion**
- Berechnet folgende Merkmalskategorien:
  - **Formmerkmale:** Aspektverhältnis, Rundheit, Exzentrizität und Hu-Momente.
  - **Farbmerkmale:** Histogramme im HSV- und Lab-Farbraum.
  - **Texturmerkmale:** GLCM-Eigenschaften, Entropie, LBP und Gabor-Filter.
  - **Kantenmerkmale:** Durchschnittliche und maximale Intensitäten von Kanten.
- Speichert Merkmale als **Textdatei (.txt)** und **NumPy-Datei (.npy)**.

**3. Vergleich von Feature-Vektoren**
- Vergleicht extrahierte Merkmalsvektoren und berechnet die Ähnlichkeit zwischen Bildern.
- Eignet sich zur quantitativen Analyse von Defekten oder Objekten.

**4. 3D-Visualisierung von Embeddings**
- Reduziert hochdimensionale Daten mittels PCA auf 3D.
- Visualisiert mehrere Datensätze simultan in einem 3D-Plot zur besseren Vergleichbarkeit.

**5. Neue Domänenanalyse**
- Ermöglicht den Vergleich eines neuen Bildes mit bestehenden Datensätzen.
- Berechnet die **euklidische Distanz** des neuen Bildes zu jedem bestehenden Datensatz.
- Zeigt das Ergebnis grafisch und numerisch an.

**6. Grafische Benutzeroberfläche (GUI)**
- Eine benutzerfreundliche Oberfläche zur Steuerung aller Hauptfunktionen:
  - Bildauswahl und Segmentierung
  - Merkmalsextraktion und Anzeige der Ergebnisse
  - Visualisierung von Embeddings
  - Vergleich neuer Daten mit bestehenden Quellen
 
----

## **Verzeichnisstruktur**

```plaintext
Projektordner/
│-- EmbeddingVisu.py           # PCA-Visualisierung von Embeddings
│-- FeatureText.py             # Merkmalsextraktion aus Bildern
│-- FeatureTextVergleich.py    # Vergleich von Feature-Vektoren
│-- Segmentation.py            # Interaktive Bildsegmentierung
│-- NewDomain.py               # Analyse neuer Domänendaten
│-- GUI.py / GUI_RUN.py        # Starten der grafischen Benutzeroberfläche
│-- TextContent.json           # GUI-Texte
│-- README.md                  # Projektbeschreibung
```

---

## **Systemanforderungen** 

Um das Projekt auszuführen, benötigen Sie:
- **Python 3.8 oder höher**
- **Betriebssystem:** Windows, Linux oder macOS
- **Python-Bibliotheken:**

```
pip install numpy opencv-python matplotlib PyQt5 scikit-image scikit-learn scipy
```

---

## **Installationsanleitung** 

**1. Repository klonen oder herunterladen:**

```
git clone <Projekt-Repository-URL>
cd Projektordner
```

**2. Abhängigkeiten installieren:**

```
pip install -r requirements.txt
```

**3. GUI starten:**
```
python GUI_RUN.py
```

---

## **Verwendung der Module** 

**1. GUI starten**
- Führen Sie GUI_RUN.py aus, um die grafische Oberfläche zu starten.

**2. Bildsegmentierung und Feature-Extraktion**
- Wählen Sie ein Bild aus und folgen Sie den Anweisungen:
  - Segmentieren Sie die relevante Region im Bild.
  - Extrahieren Sie Form-, Farb-, Textur- und Kantenmerkmale.

**3. 3D-Visualisierung von Embeddings**
- Laden Sie mehrere Verzeichnisse mit .npy-Dateien.
- Die Daten werden mittels PCA in 3D visualisiert.

**4. Neue Domänenanalyse**
- Vergleichen Sie neue Daten mit bestehenden Quellen und berechnen Sie die Abstände.

---

## **Beispielablauf** 

1. Starten der GUI:
```
python GUI_RUN.py
```
2. Bild auswählen und segmentieren.
3. Merkmale extrahieren und anzeigen.
4. Mehrere Embeddings laden und visualisieren.
5. Neue Daten vergleichen und Analyseergebnisse anzeigen.

---

## **Erklärung der Module**

- **Segmentation.py:** Ermöglicht die interaktive Bildsegmentierung.
- **FeatureText.py:** Extrahiert Merkmale (Form, Farbe, Textur, Kanten) aus einem Bild.
- **FeatureTextVergleich.py:** Vergleicht die extrahierten Merkmale.
- **EmbeddingVisu.py:** Visualisiert Embeddings in 3D mittels PCA.
- **NewDomain.py:** Vergleicht neue Daten mit bestehenden Quellen und berechnet Distanzen.
- **GUI.py/GUI_RUN.py:** Startet die grafische Benutzeroberfläche für eine benutzerfreundliche Steuerung.

---

## **Screenshots und Beispiele**

**Beispiel 1: Interaktive Bildsegmentierung**

![Interaktive Segmentierung](https://github.com/user-attachments/assets/c82757b7-b71d-46c7-b4a6-2c3214313ed4)


**Beispiel 2: 3D PCA-Visualisierung**

![3D PCA Visualisierung](https://github.com/user-attachments/assets/699869b5-bf2b-420f-ae41-b0f20c46bf66)


**Beispiel 3: Neue Domänenanalyse**

![Neue Domänenanalyse](https://github.com/user-attachments/assets/ecc2fac3-91fc-4d92-9195-07c28f6f3c2d)


---

## **Lizenz**

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der Datei LICENSE.

---

## **Kontakt**

Bei Fragen oder Feedback wenden Sie sich bitte an:

**Autor:** jingyan

**E-Mail:** uvely@student.kit.edu

**Datum:** 2024
