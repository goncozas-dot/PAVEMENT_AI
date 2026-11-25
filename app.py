import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from image_classifier import predict_image, train_from_zip


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "pavimento_simulado.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from generate_dataset import generate_dataset

        data = generate_dataset()
        data.to_csv(DATA_PATH, index=False)
        return data
    return pd.read_csv(DATA_PATH)


def condition_label(pci: float) -> str:
    if pci >= 85:
        return "Bueno"
    if pci >= 70:
        return "Satisfactorio"
    if pci >= 55:
        return "Regular"
    if pci >= 40:
        return "Pobre"
    if pci >= 25:
        return "Muy pobre"
    if pci >= 10:
        return "Serio"
    return "Fallado"


def train_model(data: pd.DataFrame):
    X = data[["antiguedad"]]
    y = data["pci_observado"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
    }
    return model, metrics


st.set_page_config(page_title="PAVEMENT AI", page_icon="🛣️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --navy: #102b46;
        --blue: #1479bd;
        --cyan: #31c3d9;
        --ink: #20364b;
        --muted: #708397;
        --surface: #ffffff;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f3f8fc 0%, #eaf1f7 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #102b46; }
    [data-testid="stSidebar"] * { color: #eef7ff; }
    [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 2rem; }
    h1 { color: var(--navy); letter-spacing: -0.04em; font-weight: 800; }
    h2, h3 { color: var(--navy); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: transparent;
        border: 0;
        border-radius: 12px;
        padding: 10px 8px;
        min-height: 88px;
        box-shadow: none;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 750;
        font-size: clamp(1rem, 1.8vw, 1.65rem);
    }
    [data-testid="stTabs"] button {
        border-radius: 10px 10px 0 0;
        font-weight: 650;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--blue);
        border-bottom-color: var(--blue);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, .92);
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(16, 43, 70, .045);
    }
    [data-testid="stForm"] {
        border: 1px solid #dce7ef;
        border-radius: 14px;
        background: white;
        padding: 1.1rem;
        box-shadow: 0 8px 24px rgba(16, 43, 70, .045);
    }
    [data-testid="stFileUploader"] section {
        border-radius: 14px;
    }
    [data-testid="stFileUploader"] section {
        min-height: 138px;
        border: 1px dashed #91aec3;
        background: #f7fbfe;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stPlotlyChart"] {
        border-radius: 14px;
        overflow: hidden;
        background: white;
    }
    [data-testid="stHorizontalBlock"] { align-items: stretch; }
    .hero {
        background: linear-gradient(120deg, #102b46, #165b87 65%, #1479bd);
        color: white;
        border-radius: 22px;
        padding: 30px 34px;
        margin: 0 0 22px;
        box-shadow: 0 16px 38px rgba(16, 43, 70, .18);
    }
    .hero h1 { color: white; margin: 0 0 6px; font-size: 2.3rem; }
    .hero p { color: #d7edfa; margin: 0; }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
        color: #72dfeb;
        margin-bottom: 8px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem; }
        .hero { padding: 22px; }
        .hero h1 { font-size: 1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">SENATI · Huánuco · Análisis vial</div>'
    "<h1>PAVEMENT AI</h1>"
    "<p>Herramientas visuales y predictivas para inspeccionar pavimento rígido.</p></div>",
    unsafe_allow_html=True,
)

data = load_data()
model, metrics = train_model(data)

tabs = st.tabs(["Dashboard", "Imágenes", "Datos", "Registro", "Metodología"])

with tabs[0]:
    training_message = st.session_state.pop("image_training_message", None)
    if training_message:
        st.success(training_message)

    with st.container(border=True):
        st.subheader("Predicción interactiva")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Antigüedad actual (años)", 1, 10, 5)
        horizon = col2.number_input("Horizonte (años)", 1, 5, 3)
        scenario = col3.selectbox(
            "Escenario", ["Base", "Bajo tránsito", "Alto tránsito"]
        )

    adjustment = {"Base": 0.0, "Bajo tránsito": 2.0, "Alto tránsito": -3.0}[scenario]
    future_age = age + horizon
    current_raw = model.predict(pd.DataFrame({"antiguedad": [age]}))[0] + adjustment
    future_raw = model.predict(
        pd.DataFrame({"antiguedad": [future_age]})
    )[0] + adjustment
    current_pci = float(np.clip(current_raw, 0, 100))
    future_pci = float(np.clip(future_raw, 0, 100))

    result_col, image_col = st.columns([1.35, 1])
    with result_col:
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("PCI actual", f"{current_pci:.1f}")
            m2.metric("PCI futuro", f"{future_pci:.1f}")
            m3.metric("Condición", condition_label(future_pci))
            st.caption("Estimación numérica basada en antigüedad y datos simulados.")

    with image_col:
        with st.container(border=True):
            st.markdown("#### Análisis por imagen")
            image_file = st.file_uploader(
                "Arrastra una foto del pavimento o haz clic para buscar",
                type=["jpg", "jpeg", "png"],
                key="dashboard_pathology_image",
                help="Selecciona una fotografía JPG o PNG de hasta 15 MB.",
            )
            classifier = st.session_state.get("image_classifier")
            if image_file is not None:
                image_bytes = image_file.getvalue()
                st.image(
                    image_bytes,
                    caption=image_file.name,
                    use_container_width=True,
                )
                if classifier is None:
                    st.info(
                        "Para clasificar la foto, primero entrena el modelo en la "
                        "pestaña Imágenes."
                    )
                else:
                    try:
                        label, probability = predict_image(classifier, image_bytes)
                        st.metric("Patología visual estimada", label)
                        st.progress(
                            probability,
                            text=f"Puntuación del clasificador: {probability:.0%}",
                        )
                        st.caption(
                            "Clasificación visual orientativa; no estima el PCI ni "
                            "reemplaza la inspección técnica."
                        )
                    except ValueError as exc:
                        st.error(str(exc))
            elif classifier is None:
                st.caption(
                    "Entrena el clasificador una vez en la pestaña Imágenes; "
                    "después podrás soltar fotos aquí."
                )
            else:
                st.caption("El clasificador está listo para analizar una fotografía.")

    with st.container(border=True):
        ages = np.arange(1, 16)
        predictions = model.predict(pd.DataFrame({"antiguedad": ages}))
        chart = pd.DataFrame(
            {
                "Antigüedad": ages,
                "PCI estimado": np.clip(predictions, 0, 100),
            }
        )
        fig = px.line(chart, x="Antigüedad", y="PCI estimado", markers=True)
        fig.add_scatter(
            x=data["antiguedad"],
            y=data["pci_observado"],
            mode="markers",
            name="Datos simulados",
        )
        fig.add_scatter(
            x=[future_age],
            y=[future_pci],
            mode="markers",
            marker={"size": 13, "color": "#d62728"},
            name="Predicción",
        )
        fig.update_layout(yaxis={"range": [0, 100]}, height=420)
        st.plotly_chart(fig, use_container_width=True)

    if future_age > 10:
        st.warning(
            "La predicción supera el rango histórico de 10 años. "
            "Es una extrapolación y debe interpretarse con cautela."
        )

    st.info(
        "Advertencia: los datos son ficticios y el ajuste por tránsito es "
        "una simulación didáctica, no un coeficiente entrenado."
    )

    with st.container(border=True):
        st.subheader("Tres escenarios")
        scenarios = []
        for name, delta in (
            ("Óptimo / bajo tránsito", 2.0),
            ("Base", 0.0),
            ("Desfavorable / alto tránsito", -3.0),
        ):
            predicted = np.clip(
                model.predict(pd.DataFrame({"antiguedad": [future_age]}))[0] + delta,
                0,
                100,
            )
            scenarios.append((name, round(float(predicted), 1), condition_label(predicted)))
        st.dataframe(
            pd.DataFrame(scenarios, columns=["Escenario", "PCI futuro", "Condición"]),
            hide_index=True,
            use_container_width=True,
        )

with tabs[1]:
    with st.container(border=True):
        st.subheader("Clasificación visual de patologías")
        st.write(
            "Carga un ZIP de fotos ordenadas en carpetas por clase para entrenar un "
            "clasificador local. Después regresa al Dashboard y arrastra allí una foto "
            "para obtener la clase visual."
        )
        st.code("dataset/grieta/foto_01.jpg\ndataset/despostillamiento/foto_02.jpg")
        st.caption(
            "Incluye al menos dos clases y dos fotos por clase. Se recomienda contar "
            "con muchas más imágenes variadas por clase para evaluar el modelo."
        )

        dataset_zip = st.file_uploader(
            "Dataset ZIP con carpetas etiquetadas",
            type=["zip"],
            key="pathology_dataset_zip",
            help=(
                "Acepta ZIP de hasta 100 MB con JPG/PNG organizados como "
                "dataset/nombre_clase/imagen.jpg."
            ),
        )
        if st.button("Entrenar clasificador de imágenes", type="primary"):
            if dataset_zip is None:
                st.error("Primero selecciona un archivo ZIP con imágenes etiquetadas.")
            else:
                try:
                    classifier, class_counts = train_from_zip(dataset_zip.getvalue())
                    st.session_state["image_classifier"] = classifier
                    st.session_state["image_class_counts"] = class_counts
                    st.session_state["image_training_message"] = (
                        f"Clasificador entrenado con {sum(class_counts.values())} imágenes "
                        f"en {len(class_counts)} clases."
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        if "image_class_counts" in st.session_state:
            counts = st.session_state["image_class_counts"]
            st.caption(
                "Clasificador listo · "
                + " · ".join(
                    f"{label}: {count}" for label, count in sorted(counts.items())
                )
            )

with tabs[2]:
    with st.container(border=True):
        st.subheader("Calidad del modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{metrics['r2']:.3f}")
        c2.metric("MAE", f"{metrics['mae']:.3f}")
        c3.metric("RMSE", f"{metrics['rmse']:.3f}")

        st.code(
            f"PCI = {model.intercept_:.2f} + "
            f"{model.coef_[0]:.2f} × antigüedad",
            language="text",
        )
        st.caption(
            "La pendiente indica cuánto cambia el PCI estimado por cada año adicional."
        )

    with st.container(border=True):
        st.subheader("Datos de entrenamiento")
        st.dataframe(data, use_container_width=True, hide_index=True)
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV",
            csv,
            file_name="pavimento_simulado.csv",
            mime="text/csv",
        )

with tabs[3]:
    st.subheader("Registro de patologías")
    with st.form("patologia_form"):
        tramo = st.text_input("Identificación del tramo")
        fecha = st.date_input("Fecha de inspección")
        antiguedad_registro = st.number_input(
            "Antigüedad (años)", min_value=0, max_value=100, value=1
        )
        transito = st.selectbox("Nivel de tránsito", ["Bajo", "Medio", "Alto"])
        patologia = st.selectbox(
            "Tipo de patología",
            [
                "Grieta longitudinal",
                "Grieta transversal",
                "Grieta de esquina",
                "Losa dividida",
                "Despostillamiento",
                "Escalonamiento",
            ],
        )
        severidad = st.selectbox("Severidad", ["Leve", "Moderada", "Alta"])
        pci_registro = st.number_input(
            "PCI observado", min_value=0.0, max_value=100.0, value=70.0
        )
        foto = st.file_uploader(
            "Fotografía", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        saved = st.form_submit_button("Agregar registro")

    if saved:
        if not tramo.strip():
            st.error("Ingrese la identificación del tramo.")
        else:
            nuevo = pd.DataFrame(
                [
                    {
                        "tramo_id": tramo.strip(),
                        "fecha_inspeccion": str(fecha),
                        "antiguedad": antiguedad_registro,
                        "nivel_transito": transito,
                        "tipo_patologia": patologia,
                        "severidad": severidad,
                        "fotografia": ", ".join(f.name for f in (foto or [])),
                        "pci_observado": pci_registro,
                        "dato_simulado": "No",
                    }
                ]
            )
            output = io.StringIO()
            nuevo.to_csv(output, index=False, header=False)
            st.session_state.setdefault("registro_extra", [])
            st.session_state["registro_extra"].append(nuevo)
            st.success("Registro validado y agregado a la sesión.")

    if st.session_state.get("registro_extra"):
        registros_extra = pd.concat(
            st.session_state["registro_extra"], ignore_index=True
        )
        with st.container(border=True):
            st.subheader("Registros agregados")
            st.dataframe(registros_extra, hide_index=True, use_container_width=True)
            st.download_button(
                "Exportar registros de la sesión",
                registros_extra.to_csv(index=False).encode("utf-8"),
                file_name="registros_pavimento.csv",
                mime="text/csv",
            )

with tabs[4]:
    with st.container(border=True):
        st.subheader("Método")
        st.markdown(
            """
            1. Se limpia el dataset y se valida el rango de antigüedad y PCI.
            2. Se separan datos de entrenamiento y prueba al azar (75 % / 25 %).
            3. Se ajusta `LinearRegression` con antigüedad como variable independiente.
            4. Se evalúan R², MAE y RMSE.
            5. La aplicación limita la predicción visualizada al rango de 0 a 100.

            **Limitaciones**
            - Antigüedad no explica por sí sola todos los daños.
            - Los datos son simulados y no representan inspecciones técnicas.
            - Fuera del rango observado, la predicción es una extrapolación.
            - El ajuste manual por tránsito no sustituye un modelo entrenado con esa variable.
            """
        )
