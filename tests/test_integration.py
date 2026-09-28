from src.data_pipeline import ingest, clean, validate, transform

def test_full_data_pipeline_runs_end_to_end():
    df = ingest()
    df = clean(df)
    assert validate(df)
    df = transform(df)
    assert "target" in df.columns
    assert len(df) > 0
