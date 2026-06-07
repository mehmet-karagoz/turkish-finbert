from turkish_fin_bert.text import clean_text, combine_title_text


def test_clean_text_removes_html_and_extra_spaces():
    assert clean_text("<b>Kar</b>   artt\u0131") == "Kar artt\u0131"


def test_clean_text_repairs_turkish_mojibake():
    original = "A.\u015e. \u00d6zel Durum A\u00e7\u0131klamas\u0131"
    broken = original.encode("utf-8").decode("cp1252")

    assert clean_text(broken) == original


def test_clean_text_treats_nan_as_empty():
    assert clean_text(float("nan")) == ""
    assert clean_text("nan") == ""


def test_combine_title_text_avoids_empty_values():
    assert combine_title_text("Ba\u015fl\u0131k", "") == "Ba\u015fl\u0131k"
    assert combine_title_text("Ba\u015fl\u0131k", "Detay metni") == "Ba\u015fl\u0131k. Detay metni"
