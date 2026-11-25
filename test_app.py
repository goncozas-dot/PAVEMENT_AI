import pandas as pd
import io
import zipfile
from PIL import Image, ImageDraw

from app import condition_label, train_model
from generate_dataset import generate_dataset
from image_classifier import predict_image, train_from_zip


def test_generated_dataset_has_expected_size():
    data = generate_dataset()
    assert len(data) >= 50
    assert data["dato_simulado"].eq("Si").all()


def test_model_metrics_are_finite():
    data = generate_dataset()
    model, metrics = train_model(data)
    assert model is not None
    assert all(metric >= 0 for metric in metrics.values())
    assert 0 <= metrics["r2"] <= 1


def test_condition_boundaries():
    assert condition_label(90) == "Bueno"
    assert condition_label(75) == "Satisfactorio"
    assert condition_label(5) == "Fallado"


def test_image_classifier_trains_on_labeled_folders():
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for label, line_color in (("grieta", 0), ("sin_dano", 255)):
            for index in range(2):
                image = Image.new("L", (64, 64), 255)
                draw = ImageDraw.Draw(image)
                if label == "grieta":
                    draw.line((0, index * 6 + 10, 63, index * 6 + 10), fill=line_color, width=3)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                archive.writestr(
                    f"dataset/{label}/foto_{index}.png", buffer.getvalue()
                )

    model, counts = train_from_zip(archive_buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(archive_buffer.getvalue())) as archive:
        prediction = predict_image(model, archive.read("dataset/grieta/foto_0.png"))

    assert counts == {"grieta": 2, "sin_dano": 2}
    assert prediction[0] == "grieta"
    assert 0 <= prediction[1] <= 1


def test_image_classifier_rejects_unlabeled_zip():
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        image = Image.new("RGB", (64, 64), "white")
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        archive.writestr("photo.png", image_buffer.getvalue())

    try:
        train_from_zip(archive_buffer.getvalue())
    except ValueError as exc:
        assert "carpetas de clases" in str(exc)
    else:
        raise AssertionError("El dataset sin etiquetas debe rechazarse.")
