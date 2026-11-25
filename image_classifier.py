import io
import zipfile
from collections import Counter
from pathlib import PurePosixPath

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 200 * 1024 * 1024
MAX_IMAGES = 2000


def image_features(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("No se pudo leer la imagen. Usa una imagen JPG o PNG válida.") from exc

    return hog(
        pixels,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def train_from_zip(archive_bytes: bytes):
    if not archive_bytes or len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("El ZIP está vacío o supera el límite de 100 MB.")

    features = []
    labels = []
    total_image_bytes = 0

    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("El archivo no es un ZIP válido.") from exc

    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("El ZIP contiene una ruta de archivo no permitida.")
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if len(path.parts) == 1:
                raise ValueError(
                    "Cada imagen debe estar dentro de una carpeta de clase. "
                    "Incluye al menos dos carpetas de clases en el ZIP."
                )
            if len(path.parts) not in (2, 3):
                raise ValueError(
                    "Organiza las fotos directamente en carpetas de clase: "
                    "dataset/grieta/foto.jpg."
                )
            if member.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"La imagen {path.name} supera el límite de 15 MB.")

            total_image_bytes += member.file_size
            if total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("Las imágenes del ZIP superan el límite total de 200 MB.")
            if len(features) >= MAX_IMAGES:
                raise ValueError("El ZIP supera el límite de 2000 imágenes.")

            label = path.parts[-2].strip()
            if not label:
                raise ValueError(f"No se encontró una etiqueta de clase para {path.name}.")

            try:
                image_bytes = archive.read(member)
                features.append(image_features(image_bytes))
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(f"No se pudo procesar {path.as_posix()}: {exc}") from exc
            labels.append(label)

    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("El ZIP debe incluir al menos dos carpetas de clases.")
    small_classes = [label for label, count in counts.items() if count < 2]
    if small_classes:
        raise ValueError(
            "Cada clase necesita al menos dos imágenes. Revisa: "
            + ", ".join(sorted(small_classes))
        )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    )
    model.fit(np.stack(features), np.asarray(labels))
    return model, dict(counts)


def predict_image(model, image_bytes: bytes) -> tuple[str, float]:
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("La fotografía está vacía o supera el límite de 15 MB.")
    probabilities = model.predict_proba(image_features(image_bytes).reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    return str(model.classes_[best_index]), float(probabilities[best_index])
