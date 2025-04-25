import os
import pytest
import subprocess

PIPELINE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline.py'))

@pytest.mark.parametrize("step,expected", [
    ('priors', ''),
])
def test_pipeline_priors(step, expected, tmp_path, monkeypatch, capsys):
    # Monkeypatch run_script to simulate successful runs
    monkeypatch.setenv('PYTHONPATH', os.getcwd())
    # Run the priors step
    result = subprocess.run(
        ['python', PIPELINE, step, '-v'],
        cwd=os.getcwd(), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    # Check that output is empty for priors step
    assert result.stdout == expected

# More tests can be added for summary and figures steps, using monkeypatch to disable actual script execution

# Additional tests for data validation

def test_load_states_success(tmp_path):
    data_dir = tmp_path / '_data'
    data_dir.mkdir()
    csv = data_dir / 'states_and_regions.csv'
    csv.write_text(",region,state\n0,south1,lagos\n1,south2,oyo\n")
    import pipeline
    states = pipeline.load_states(str(data_dir))
    assert set(states) == {'lagos', 'oyo'}

def test_load_states_file_not_found(tmp_path):
    import pipeline
    with pytest.raises(FileNotFoundError):
        pipeline.load_states(str(tmp_path / '_data'))

def test_load_states_missing_columns(tmp_path):
    data_dir = tmp_path / '_data'
    data_dir.mkdir()
    csv = data_dir / 'states_and_regions.csv'
    csv.write_text("foo,bar\n1,2\n")
    import pipeline
    with pytest.raises(ValueError):
        pipeline.load_states(str(data_dir))

def test_load_states_no_states(tmp_path):
    data_dir = tmp_path / '_data'
    data_dir.mkdir()
    csv = data_dir / 'states_and_regions.csv'
    csv.write_text("region,state\nnorth1,alpha\nnorth2,beta\n")
    import pipeline
    with pytest.raises(ValueError):
        pipeline.load_states(str(data_dir))

def test_generate_figures_invalid_state(monkeypatch, tmp_path):
    import pipeline
    monkeypatch.setattr(pipeline, 'load_states', lambda x: ['lagos'])
    with pytest.raises(ValueError):
        pipeline.generate_figures('oyo', data_dir=str(tmp_path / '_data'))
