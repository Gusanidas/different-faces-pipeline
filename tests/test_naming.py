import pytest

from different_pipeline_generator.naming import clean_comfy_name, ensure_unique


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("c005_s3_00001_.png", "c005_s3"),
        ("c005_s3_00002.png", "c005_s3"),
        ("nested/c005_s3_01234_.webp", "nested%2Fc005_s3"),
        ("already_clean.png", "already_clean"),
    ],
)
def test_clean_comfy_name(filename, expected):
    assert clean_comfy_name(filename) == expected


def test_duplicate_names_raise_before_rendering():
    with pytest.raises(ValueError, match="duplicate identities"):
        ensure_unique(["person_a", "person_b", "person_a"], label="identities")


def test_duplicate_basenames_in_different_directories_stay_unique():
    assert clean_comfy_name("bank_a/face_00001_.png") != clean_comfy_name(
        "bank_b/face_00001_.png"
    )
