import text_processing_demo


def test_version():
    assert text_processing_demo.__version__ == "0.1.0"


def test_clean_overview_text_first_sentence_only():
    # Given
    overview = (
        "this is the first sentence. "
        "this is the second sentence. "
        "this is the third..."
    )

    # When
    result = text_processing_demo.clean_overview_text(overview)

    # Then
    expected = "first sentence"
    assert result == expected


def test_clean_overview_text_first_sentence_exclamation():
    # Given
    overview = (
        "this is the first sentence! "
        "this is the second sentence. "
        "this is the third..."
    )

    # When
    result = text_processing_demo.clean_overview_text(overview)

    # Then
    expected = "first sentence"
    assert result == expected


def test_clean_overview_text_first_sentence_question():
    # Given
    overview = (
        "this is the first sentence? "
        "this is the second sentence. "
        "this is the third..."
    )

    # When
    result = text_processing_demo.clean_overview_text(overview)

    # Then
    expected = "first sentence"
    assert result == expected


def test_clean_overview_text_no_punctuation():
    # Given
    overview = "this is a great-looking, exceptional, film."

    # When
    result = text_processing_demo.clean_overview_text(overview)

    # Then
    expected = "greatlooking exceptional film"
    assert result == expected


def test_clean_overview_text_lowercase():
    # Given
    overview = "This IS tHe fIRst SENTENCE."

    # When
    result = text_processing_demo.clean_overview_text(overview)

    # Then
    expected = "first sentence"
    assert result == expected
