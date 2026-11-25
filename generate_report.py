from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent


def generate_report() -> None:
    data = pd.read_csv(ROOT / "pavimento_simulado.csv")
    X = data[["antiguedad"]]
    y = data["pci_observado"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)
    rmse = mean_squared_error(y_test, predicted) ** 0.5

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(ROOT / "informe_PAVEMENT_AI.pdf"),
        pagesize=A4,
        title="Informe PAVEMENT AI",
    )
    story = [
        Paragraph("PAVEMENT AI", styles["Title"]),
        Paragraph("Informe de desarrollo y resultados", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(
            "Proyecto individual SENATI - Predicción del deterioro del pavimento rígido "
            "mediante regresión lineal.",
            styles["BodyText"],
        ),
        Paragraph("Resumen", styles["Heading2"]),
        Paragraph(
            f"Se utilizaron {len(data)} registros simulados en seis tramos. "
            "La variable independiente fue antigüedad y la dependiente fue el PCI observado.",
            styles["BodyText"],
        ),
        Paragraph("Modelo", styles["Heading2"]),
        Paragraph(
            f"Modelo obtenido: PCI = {model.intercept_:.2f} + "
            f"({model.coef_[0]:.2f} × antigüedad).",
            styles["BodyText"],
        ),
        Paragraph("Métricas de prueba", styles["Heading2"]),
    ]

    metrics = [
        ["Métrica", "Valor"],
        ["R²", f"{r2_score(y_test, predicted):.3f}"],
        ["MAE", f"{mean_absolute_error(y_test, predicted):.3f}"],
        ["RMSE", f"{rmse:.3f}"],
    ]
    table = Table(metrics, colWidths=[120, 120])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123e65")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#edf4fa")),
            ]
        )
    )
    story.append(table)
    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Conclusiones", styles["Heading2"]),
            Paragraph(
                "La antigüedad presenta una relación negativa con el PCI. "
                "La pendiente indica el cambio promedio estimado del PCI por año adicional.",
                styles["BodyText"],
            ),
            Paragraph(
                "La antigüedad no sustituye una evaluación técnica; el tránsito, clima, "
                "mantenimiento y severidad pueden explicar variaciones no captadas por el modelo.",
                styles["BodyText"],
            ),
            Paragraph("Nota", styles["Heading2"]),
            Paragraph(
                "Todos los datos y resultados son simulados con fines educativos y no representan "
                "una inspección real ni una condición estructural certificada.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story)


if __name__ == "__main__":
    generate_report()
