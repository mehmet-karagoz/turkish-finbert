from turkish_fin_bert.text import clean_text, combine_title_text


def test_clean_text_removes_html_and_extra_spaces():
    assert clean_text("<b>Kar</b>   arttı") == "Kar arttı"


def test_clean_text_treats_nan_as_empty():
    assert clean_text(float("nan")) == ""
    assert clean_text("nan") == ""


def test_combine_title_text_avoids_empty_values():
    assert combine_title_text("Başlık", "") == "Başlık"
    assert combine_title_text("Başlık", "Detay metni") == "Başlık. Detay metni"
