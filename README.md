# Ziele

Im Rahmen unserer Projektarbeit für Learning und Softcomputing haben wir uns mit einer Sentiment-Analyse von zwei verschiedenen Datenquellen beschäftig.

Unsere selbst gestecken Ziele für die Projektarbeit lassen sich wie folgt zusammenfassen:
- Wir wollten verschiedene NLP-Ansätze ausprobieren, um die Sentiment-Klassen der Daten vorherzusagen. Dabei haben wir uns vorgenommen verschiedene klassische Machine Learning-Ansätze ohne Deep Learning sowie einen Deep Learning-Ansatz auszuprobieren und miteinander zu vergleichen.
- Im Rahmen der verschiedenen Techniken wollten wir außerdem, besonders bei dem Deep Learning-Ansatz, die Auswirkung von unterschiedlichen Hyperparametern evaluieren.
- Neben der Variation an Verfahren haben wir zwei unterschiedliche Datensätze (Movie Reviews und Tweets) mit den gleichen Techniken bearbeitet und auch diese Ergebnisse miteinander verglichen.
- Neben der inhaltlichen Evaluation war ein weiteres wichtiges Ziel für uns, viel praktische Erfahrung in Bezug auf das methodische Vorgehen bei solchen Data Science Projekten zu sammeln.

# Aufbau der Projektdokumentation

Da im Rahmen der Projektarbeit eine Menge an Code entstanden ist, haben wir uns für eine gemischte Nutzung von Colab Notebooks und normalen Python Skripten entschieden. Gerade bei dem Training der Neuronalen Netze haben wir einige Klassen geschrieben, die wir lediglich in Notebooks importieren, um Sie dann auf Colab auszuführen.

Der relevante Ergebnisteil des Projektarbeit ist jedoch komplett in Notebooks verfasst. Dazu haben wir im Ordner 'Notebooks' 3 relevante Notebooks angelegt. Wir empfehlen zur Nachverfolgung unseres Projektes dabei diese Reihenfolge:
1. *ExploratoryDataAnalysis.ipynb*: Hier werden die beiden genannten Datensätze explorativ untersucht und einen Teil des Preprocessing erledigt
2. *klassische_Klassifikationsmodelle.ipynb*: In diesem Notebook haben wir diverse klassische Klassifikationsalgorithmen (keine ANNs) auf unsere Daten angewandt und mit unterschiedlichen Parametern sowie Arten der Feature-Erstellung getestet, was die bestmögliche Accuracy liefert. 
3. *large_language_models.ipynb*: In diesem Notebook sind das gesamte Vorgehen beim Training von Neuronalen Netzen sowie die Ergebnisse zusammengefasst und präsentiert (wir raten davon ab, diese Notebook auszuführen, da die meisten Auswertungen ohne GPU sehr lange brauchen).

# Gesamtfazit

In den Notebook 2. und 3. finden sich jeweils spezifische Fazits zu den einzelnen Techniken. Im Hinblick auf die oben genannten Ziele können wir hier aber folgende übergreifenden Erkenntnisse zusammenfassen:
- Im Vergleich zu den klassischen Ansätzen ist der Deep Learning-Ansatz mit der Datenmenge und der Rechenleistung, die uns für dieses Projekt zur Verfügung stand, nicht deutlich besser. Nur mit einem Trick (Early Stopping) konnte auf dem Movie Review-Datensatz eine um 0.06 bessere Accuracy erreicht werden.
- Bei beiden Ansätzen ist die Performance bei dem Tweets-Datensatz deutlich höher. Dies kann an der reduzierten Klassenanzahl, an dem deutlich größeren Umfang des Datensatzes oder an der Domäne liegen.
- Im Vergleich bzgl. der Übertragbarkeit zwischen den Domänen zeigt sich ein gemischtes Bild. Bei den Modellen, die auf dem Movie Review-Datensatz trainiert wurden, sind die klassischen Modelle etwas besser übertragbar auf den anderen Datensatz. Bei dem Tweets-Modellen liegt das Deep Learning-Ansatz ein wenig vorne.
- Bei kleinen Datensätzen lohnt es sich immer zunächst einfache/klassische Modell auszuprobieren, weil Deep Learning-Modell mehr Aufwand bedeuten, mehr Rechenkapazität benötigen und mehr Daten benötigen, um gut lernen zu können.
- Beim Training von Neuronalen Netzen ist es sinnvoll sich vorher alle möglichen Trainingsvarianten zu überlegen im Hinblick auf das Datenformat, das Preprocessing oder das Hyperparametertuning, um von Anfang an eine sinnvolle Programmstruktur zu nutzen.
- Das Übertragen von verschiedenen Klassifizierungsmodellen funktioniert nur sehr bedingt, wenn die Modelle und die entsprechenden Trainingsdaten eine unterschiedliche Anzahl an Klassen haben. Obwohl die Klassen bei uns ordinal bis annähernd metrisch sind hat die Vergleichbarkeit darunter gelitten. Hauptursache ist unserer Einschätzung nach, dass die eine Skala einen Mittelpunkt hat (neutral), die andere aber nicht.