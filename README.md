# Grain Analyzer

A standalone package for performing grain analysis on XQD files.

## Features

- Grain detection and analysis from XQD files
- Extraction of individual grain data
- Calculation of grain statistics
- Generation of PDF containing original and grain_mask

## Installation

### From Source

Run the following command in the root directory of the project:

```bash
pip install .
```

### From GitHub

You can install directly from GitHub:

```bash
pip install git+https://github.com/jkkwoen/grain_analyzer.git
```

### For Development

If you plan to modify the code, install in editable mode:

```bash
# Create virtual environment (Optional)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install in editable mode
pip install -e .
```

## Usage

### Using as a Python script

```python
from pathlib import Path
from grain_analyzer.analyze import analyze_single_file_with_grain_data

xqd_file = Path("path/to/your/file.xqd")
output_dir = Path("output")

success, individual_grain_data, grain_stats, pdf_path = analyze_single_file_with_grain_data(
    xqd_file, output_dir
)

if success:
    print(f"PDF saved: {pdf_path}")
    print(f"Number of grains: {grain_stats['num_grains']}")
    print(f"First grain area: {individual_grain_data[0]['area_nm2']} nm²")
```

### Importing directly

```python
from pathlib import Path
from grain_analyzer import analyze_single_file_with_grain_data

# or
from grain_analyzer.analyze import analyze_single_file_with_grain_data
```

## Output

- **PDF File**: Plot including original height data and grain_mask overlay
- **Individual Grain Data**: Detailed information for each grain (area, diameter, centroid, peak position, etc.)
- **Grain Statistics**: Statistical information for all grains

## Dependencies

- Python 3.8+
- numpy>=1.24.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- scikit-image>=0.20.0


## Project Structure

```
grain_analyzer/
├── grain_analyzer/
│   ├── __init__.py
│   ├── io.py               # Read XQD files
│   ├── corrections.py      # Correction functions
│   ├── grain_analysis.py   # Grain analysis functions
│   ├── utils.py            # Utility functions
│   ├── afm_data_wrapper.py # AFMData wrapper class
│   └── analyze.py          # Main analysis function
├── requirements.txt
├── setup.py
└── README.md
```
