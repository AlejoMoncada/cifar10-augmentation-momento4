"""Build the Momento 4 comparative report PDF in the Act3 visual format."""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"
FIGS = ROOT / "report" / "figures"
OUT = ROOT / "report" / "Act4_Reporte_Comparativo_Aumentacion.pdf"
REPO = "https://github.com/AlejoMoncada/cifar10-augmentation-momento4"

for name in ("Regular", "Bold", "Italic", "BoldItalic"):
    pdfmetrics.registerFont(TTFont(f"Carlito-{name}", str(ASSETS / f"Carlito-{name}.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
pdfmetrics.registerFontFamily("Carlito", normal="Carlito-Regular", bold="Carlito-Bold",
                              italic="Carlito-Italic", boldItalic="Carlito-BoldItalic")

BODY = ParagraphStyle("body", fontName="Carlito-Regular", fontSize=9.6, leading=12.2, alignment=TA_JUSTIFY,
                      spaceAfter=4)
HEAD = ParagraphStyle("head", fontName="Arial-Bold", fontSize=9.5, leading=12, spaceAfter=5)
CENTER_HEAD = ParagraphStyle("chead", parent=HEAD, alignment=TA_CENTER, spaceBefore=6, spaceAfter=12)
TITLE = ParagraphStyle("title", fontName="Arial-Bold", fontSize=17, leading=20.5, spaceBefore=4, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12.5, leading=15, spaceBefore=7, spaceAfter=4)
CAPTION = ParagraphStyle("cap", fontName="Carlito-Italic", fontSize=8, leading=10, alignment=TA_CENTER,
                         textColor=colors.HexColor("#595959"), spaceBefore=2, spaceAfter=6)
CELL = ParagraphStyle("cell", fontName="Carlito-Regular", fontSize=8, leading=9.6)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Carlito-Bold", alignment=TA_CENTER)
CELL_C = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)

W = letter[0] - 4.4 * cm


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return ListFlowable([ListItem(p(t), leftIndent=12, value="•") for t in items], bulletType="bullet",
                        start="•", leftIndent=14, bulletFontSize=8)


def table(rows, widths, center_cols=()):
    data = [[p(str(c), CELL_B) for c in rows[0]]]
    for r in rows[1:]:
        data.append([p(str(c), CELL_C if i in center_cols else CELL) for i, c in enumerate(r)])
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def figure(path, width, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), p(caption, CAPTION)])


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceBefore=4, spaceAfter=6)


def pct(v):
    return f"{v * 100:.2f}".replace(".", ",")


def pp(v):
    return f"{v:+.2f}".replace(".", ",")


def build():
    comp = pd.read_csv(ROOT / "results" / "scenario_comparison.csv").set_index("scenario")
    gaps = pd.read_csv(ROOT / "results" / "generalization_gap_report.csv").set_index("scenario")
    cls = pd.read_csv(ROOT / "results" / "single_transform_classification.csv").set_index("scenario")
    order = ["reference", "full", "flip_h", "rotation", "zoom", "flip_v", "blur"]

    s = []
    s.append(Image(str(ASSETS / "logo_sabana.png"), width=4.2 * cm, height=4.2 * cm * 192 / 447))
    s.append(p("MAESTRIA EN INTELIGENCIA ARTIFICIAL", CENTER_HEAD))
    s.append(p("Asignatura: Técnicas Avanzadas de Modelado en IA", HEAD))
    s.append(p("Actividad Formativa - Calidad de datos visuales e interpretación: aumentación controlada", HEAD))
    s.append(p("Etapa: Unidad 2", HEAD))
    s.append(p("Integrantes: Edgar Julian Mendez Ortegon - Noel Eduardo Perez Barrios - William "
               "Aljandro Moncada Cifuentes", HEAD))
    s.append(rule())
    s.append(p("Actividad: Aumentación controlada de datos en CIFAR-10", TITLE))
    s.append(rule())
    s.append(p("El objetivo es comparar un escenario de referencia sin aumentación con escenarios de aumentación "
               "controlada, aplicada <b>solo al entrenamiento</b>, manteniendo constantes la arquitectura, las "
               "particiones y el presupuesto de entrenamiento del Momento 3. La evaluación se realiza sobre el "
               f"conjunto de prueba oficial sin transformar. Código, notebook ejecutado y resultados: <b>{REPO}</b>."))

    s.append(p("Diagnóstico del conjunto de datos", H2))
    s.append(table([
        ["Aspecto", "Evidencia", "Implicación"],
        ["Resolución", "32 × 32 × 3, uint8 [0, 255] → float32 [0, 1]", "Poco detalle por objeto; transformaciones fuertes borran señal"],
        ["Balance", "4.000 / 1.000 / 1.000 imágenes por clase (train / val / test)", "Comparación sin sesgo por desbalance"],
        ["Nitidez (var. Laplaciano)", "Automóvil 3.346 y Camión 3.271; Pájaro 1.817 y Ciervo 1.877", "Dificultad visual distinta entre clases"],
        ["Variabilidad", "Brillo: Avión 0,559, Rana 0,417; mayor dispersión en Automóvil y Gato", "Clases con distinta variabilidad interna"],
        ["Duplicados (pHash ≤ 4)", "117 pares candidatos train ↔ test, todos de la misma clase", "Candidatos, no duplicados confirmados"],
    ], [0.2, 0.46, 0.34]))
    s.append(Spacer(1, 4))
    s.append(p("El conjunto utilizado mantiene las características originales de CIFAR-10. Las particiones se encuentran "
               "balanceadas, lo que permite que la comparación entre escenarios no esté condicionada por diferencias de "
               "representación entre clases. Las estadísticas descriptivas muestran que las clases presentan "
               "características visuales distintas, lo que indica que el problema no presenta la misma dificultad para "
               "todas las categorías aun cuando exista balance en número de ejemplos. Los 117 pares pHash representan "
               "únicamente similitud perceptual; para confirmarlos sería necesario complementar con inspección visual, "
               "comparación directa de píxeles o SSIM, y verificar etiquetas y procedencia."))

    s.append(p("Selección de transformaciones", H2))
    s.append(table([
        ["Condición", "Transformación", "Tipo", "Justificación respecto a la semántica de las clases"],
        ["reference", "Ninguna (capa identidad)", "—", "Control; misma CNN del Momento 3 sin el bloque de aumento"],
        ["full", "Flip horizontal + rotación 0,1 + zoom 0,1", "Geométrica", "Bloque exacto del Momento 3"],
        ["flip_h", "RandomFlip horizontal", "Geométrica", "Conserva la identidad de animales y vehículos"],
        ["rotation", "RandomRotation(0,1): ±36°", "Geométrica", "Prueba tolerancia a orientación; ángulos altos son poco frecuentes"],
        ["zoom", "RandomZoom(0,1): ±10 %", "Geométrica", "Prueba variación de escala"],
        ["flip_v", "RandomFlip vertical", "Geométrica", "Sonda: objetos invertidos no pertenecen a la distribución real"],
        ["blur", "Desenfoque gaussiano, σ 0,5–1,5", "Fotométrica", "Sonda: puede eliminar detalle discriminante a 32 × 32"],
    ], [0.12, 0.28, 0.13, 0.47]))
    s.append(Spacer(1, 4))
    s.append(figure(FIGS / "aug_strip.png", W * 0.92,
                    "Figura 1. Las mismas imágenes de entrenamiento bajo cada condición (una realización aleatoria)."))

    s.append(p("Protocolo equivalente", H2))
    s.append(p("Todas las condiciones comparten la CNN del Momento 3 (tres bloques convolucionales con Batch Normalization, "
               "Max Pooling y Dropout de 0,25, 0,30, 0,40 y 0,50; <b>290.090 parámetros</b>), la partición estratificada "
               "80/20 con <b>random_state=42</b>, normalización /255, Adam (lr = 1e-3), entropía cruzada categórica, lotes "
               "de 128, máximo 30 épocas, EarlyStopping (val_loss, paciencia 6, restauración de pesos) y ReduceLROnPlateau "
               "(factor 0,5, paciencia 2). La única diferencia es el bloque de aumentación, incorporado dentro del modelo. "
               "El notebook incluye una aserción para las siete condiciones que comprueba que "
               "<b>augmentation(sample, training=False)</b> devuelve exactamente la imagen original; por tanto, validación "
               "y prueba permanecen sin transformar. A diferencia del Momento 3 (Colab con GPU y FP16), la ejecución se "
               "realizó en CPU con precisión float32, igual para todas las condiciones."))

    s.append(p("Resultados comparativos", H2))
    rows = [["Condición", "Exactitud (%)", "F1 macro (%)", "Δ exactitud (pp)", "Δ F1 (pp)", "Brecha (pp)", "Épocas", "Clasificación"]]
    for sc in order:
        r, g = comp.loc[sc], gaps.loc[sc]
        label = "control" if sc == "reference" else ("bloque M3" if sc == "full" else cls.loc[sc, "classification"])
        rows.append([sc, pct(r["accuracy"]), pct(r["macro_f1"]), pp(r["delta_accuracy_pp"]), pp(r["delta_macro_f1_pp"]),
                     pp(g["generalization_gap_pp"]), int(g["epochs_trained"]), label])
    s.append(table(rows, [0.12, 0.12, 0.12, 0.13, 0.11, 0.11, 0.08, 0.21], center_cols=range(1, 8)))
    s.append(p("Tabla 1. Desempeño en prueba (seed 42). Brecha = exactitud de entrenamiento − validación en la mejor época. "
               "Clasificación con umbral de ±0,5 pp de F1 macro frente a la referencia.", CAPTION))
    s.append(figure(FIGS / "curves_bars.png", W,
                    "Figura 2. Exactitud de validación por época y F1 macro en prueba por condición."))

    s.append(p("Análisis de generalización", H2))
    s.append(p("El aumento completo no mejoró el desempeño respecto a la referencia: <b>reference</b> obtuvo 83,75 % de "
               "exactitud y 83,72 % de F1 macro, mientras que <b>full</b> alcanzó 74,06 % y 73,52 %, una reducción de 9,69 y "
               "10,20 puntos porcentuales. Un aspecto importante para interpretar este resultado es que la referencia ya "
               "mostraba poco sobreajuste: en su mejor época la exactitud de entrenamiento fue 85,99 % y la de validación "
               "83,89 %, una brecha de apenas +2,10 pp. No existía una divergencia importante que hiciera necesario "
               "introducir una regularización mucho más fuerte."))
    s.append(p("Además, la arquitectura ya incorporaba Batch Normalization y cuatro niveles de Dropout. Al introducir el bloque "
               "<b>full</b> se incrementó la dificultad y la regularización efectiva del entrenamiento, pero se mantuvo el mismo "
               "presupuesto de 30 épocas. Las curvas lo reflejan: <b>reference</b> alcanza cerca de 86 % de exactitud de "
               "entrenamiento, mientras que <b>full</b> llega a cerca de 73 % y su validación se queda alrededor del 75 %. El "
               "comportamiento es más consistente con un modelo que no alcanzó a ajustar las representaciones con el mismo "
               "presupuesto que con la corrección de un sobreajuste previo. La brecha de <b>full</b> fue incluso negativa "
               "(−1,94 pp), pero esto no significa que generalice mejor: durante el entrenamiento el modelo observa imágenes "
               "transformadas, mientras que la validación usa imágenes originales. Una brecha pequeña o negativa debe "
               "interpretarse junto con la exactitud y el F1, no como evidencia aislada de mejor generalización."))
    s.append(p("Existe además coherencia con el Momento 3: allí la CNN con el bloque de aumento obtuvo 74,55 % de exactitud y "
               "F1 macro de 0,7439; aquí <b>full</b>, que recupera ese bloque, obtuvo 74,06 % y 0,7352. La diferencia de "
               "0,49 pp respalda la consistencia del comportamiento observado entre ambas actividades."))

    s.append(p("Errores y análisis por clase", H2))
    s.append(figure(FIGS / "confusion_delta.png", W,
                    "Figura 3. Matrices de confusión normalizadas (referencia y full) y cambio de F1 por clase."))
    s.append(p("El deterioro de <b>full</b> se distribuye entre todas las clases. Las mayores reducciones de F1 aparecen en "
               "Gato (−14,96 pp), Perro (−14,95 pp), Rana (−14,54 pp), Ciervo (−13,20 pp) y Pájaro (−11,20 pp); Automóvil y "
               "Barco pierden alrededor de 4 pp. Con <b>full</b>, el 21,9 % de los gatos se clasifica como rana, el 14,9 % de los "
               "perros como gato, el 13,2 % de los pájaros como rana y el 12,1 % de los ciervos como rana: un incremento de "
               "errores entre categorías animales visualmente próximas, relevante dada la resolución de 32 × 32. Algunos "
               "errores se realizan con confianza superior al 97–99 % (por ejemplo Gato → Rana o Automóvil → Camión) y "
               "requieren inspección individual, ya que pueden corresponder a imágenes ambiguas, pérdida de detalle o "
               "problemas de etiquetado. Una probabilidad alta no garantiza que la predicción sea correcta."))

    s.append(p("Transformaciones útiles, neutras y perjudiciales", H2))
    s.append(p("Ninguna transformación individual supera el umbral de +0,5 pp de F1 macro para considerarse útil. "
               "<b>flip_h</b> es la que mejor conserva el desempeño (83,82 % frente a 83,72 %, +0,09 pp) y se clasifica como "
               "neutra. Las demás lo reducen: <b>zoom</b> −3,23 pp, <b>rotation</b> −7,11 pp, <b>flip_v</b> −7,83 pp y "
               "<b>blur</b> −33,08 pp; este último activa el paro temprano en la época 8 con su mejor validación en la época 2. "
               "Los resultados son coherentes con los riesgos semánticos planteados: el reflejo horizontal mantiene la "
               "identidad de animales y vehículos, el vertical genera objetos invertidos poco representativos, el desenfoque "
               "elimina buena parte de la poca información espacial disponible y las rotaciones de hasta ±36° introducen "
               "orientaciones poco frecuentes. La ablación apoya la interpretación de <b>full</b>: no toda aumentación es "
               "perjudicial, ya que <b>flip_h</b> conserva la referencia; el deterioro aparece al incorporar transformaciones "
               "que aumentan progresivamente la dificultad del problema."))

    s.append(p("Conclusiones y limitaciones", H2))
    s.append(p("Bajo la arquitectura, partición, hiperparámetros y presupuesto evaluados, un bloque de aumento más agresivo no "
               "produjo una mejora de generalización. La referencia ya presentaba una brecha reducida (+2,10 pp) y la CNN "
               "incorporaba Batch Normalization y Dropout progresivo, por lo que no mostraba un sobreajuste severo que la "
               "aumentación necesitara corregir. Al agregar transformaciones sin aumentar el presupuesto de 30 épocas, la "
               "exactitud de entrenamiento de <b>full</b> cayó junto con la de validación y prueba. No puede afirmarse que la "
               "aumentación sea perjudicial en general, sino que la combinación e intensidad utilizadas no resultaron "
               "beneficiosas bajo estas condiciones; su selección debe considerar tanto la semántica de las clases como la "
               "resolución disponible. La aumentación de datos complementa, pero no reemplaza, un conjunto representativo."))
    s.append(bullets([
        "<b>Una sola semilla (42):</b> las diferencias son descriptivas de esta ejecución; el cambio de +0,09 pp de flip_h está dentro del ruido esperable.",
        "<b>Presupuesto de 30 épocas:</b> ninguna condición salvo blur activó el paro temprano; las condiciones con aumentación podrían requerir más épocas u otros hiperparámetros.",
        "<b>Resolución y duplicados:</b> CIFAR-10 es de 32 × 32 y los 117 pares pHash no están confirmados como duplicados.",
        "<b>Trabajo posterior:</b> repetir con varias semillas (media, desviación e intervalos de confianza), ampliar el presupuesto para las condiciones aumentadas y evaluar intensidades menores de rotación, zoom y blur.",
    ]))
    s.append(p("Referencias", H2))
    s.append(p("TensorFlow Developers. (2024). <i>Data augmentation</i>. https://www.tensorflow.org/tutorials/images/data_augmentation · "
               "TensorFlow Developers. (2024). <i>Image classification</i>. https://www.tensorflow.org/tutorials/images/classification · "
               "Zhang, A., Lipton, Z. C., Li, M., &amp; Smola, A. J. (2023). <i>Image Augmentation</i>. Dive into Deep Learning. "
               "https://d2l.ai/chapter_computer-vision/image-augmentation.html · Krizhevsky, A. (2009). <i>Learning Multiple "
               "Layers of Features from Tiny Images</i>. University of Toronto.",
               ParagraphStyle("ref", parent=BODY, fontSize=8.3, leading=10.2)))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Reporte comparativo - Actividad 4 - Aumentación controlada de datos en CIFAR-10",
                            author="Edgar Julian Mendez Ortegon; Noel Eduardo Perez Barrios; William Aljandro Moncada Cifuentes",
                            subject="Técnicas Avanzadas de Modelado en IA - Unidad 2")
    doc.build(s)
    print("written", OUT)


if __name__ == "__main__":
    build()
