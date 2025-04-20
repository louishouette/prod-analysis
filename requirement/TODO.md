# Truffle Production Forecasting Application Restructuring

## Primary Objective
Create a modular and maintainable application for forecasting truffle production in upcoming seasons using multiple independent statistical models.

## Core Requirements

1. **Purpose**: The application must accurately forecast the next season's truffle production.

2. **Model Independence**: All forecasting models must operate independently, with no cross-model dependencies.

3. **Data Sources**:
   - Primary input: `production.csv` - Historical production data
   - Secondary input: `ramp-up.csv` - Production targets by tree age

4. **Dual Output Requirement**: Each model must provide:
   - A forecast based solely on `production.csv`
   - A calibrated forecast that incorporates `ramp-up.csv` data

5. **Standardized Output**: All model outputs must conform to a consistent format for easy comparison and aggregation.

6. **Clear Responsibility Separation**:
   - Main program: Handles file I/O, output naming, and result organization
   - Model modules: Manage data transformations and statistical calculations

## Architecture Guidelines

### Directory Structure
```
truffle_forecast/
├── models/                 # All statistical models
│   ├── __init__.py         # Model registration
│   ├── base.py             # Base model interface
│   ├── gompertz.py         # Gompertz growth model
│   ├── state_space.py      # State space model
│   ├── exponential.py      # Exponential smoothing model
│   ├── linear_trend.py     # Linear trend model
│   ├── holts_method.py     # Holt's trend method model
│   └── utils/              # Shared utilities
│       ├── __init__.py
│       ├── data.py         # Data preparation functions
│       └── math.py         # Math utilities (growth functions, etc.)
├── output/                 # Generated outputs
│   ├── forecasts/          # Forecast results
│   ├── plots/              # Visualization outputs
│   └── reports/            # Summary reports
├── data/                   # Data directory
│   ├── input/              # Raw input data
│   └── processed/          # Processed datasets
└── main.py                 # Main application entry point
```

### Model Interface
Each model must implement a standard interface:

```python
class ForecastModel:
    """Base model interface for all forecasting models"""
    
    def __init__(self, name):
        self.name = name
    
    def fit(self, production_data):
        """Fit model using only production data"""
        pass
    
    def fit_calibrated(self, production_data, rampup_data):
        """Fit model using production data and ramp-up data for calibration"""
        pass
    
    def forecast(self, horizon=1):
        """Generate forecast for the specified time horizon"""
        pass
    
    def get_metrics(self):
        """Return model performance metrics"""
        pass
    
    def get_forecast_dataframe(self):
        """Return forecast results as a standardized DataFrame"""
        pass
```

### Standard Output Format
All model forecasts should be returned as DataFrames with the following columns:
- `Saison`: Season identifier (e.g., "2025-2026")
- `Parcelle`: Parcel identifier
- `Age`: Tree age in years
- `Forecast_Production`: Forecasted production in grams
- `Lower_CI`: Lower confidence interval (if applicable)
- `Upper_CI`: Upper confidence interval (if applicable)
- `Model`: Model identifier
- `Calibrated`: Boolean indicating if ramp-up calibration was used

## Implementation Tasks

1. **Create Base Framework**:
   - Implement the model interface
   - Create data loading and validation utilities
   - Set up the output structure

2. **Refactor Existing Models**:
   - Migrate each model to its own module
   - Ensure all models implement the standard interface
   - Remove any cross-model dependencies

3. **Standardize Data Flow**:
   - Create unified data preparation functions
   - Implement consistent error handling
   - Ensure proper encoding management

4. **Develop Main Program**:
   - Create model registry and discovery mechanism
   - Implement orchestration logic
   - Add command-line interface for customization

5. **Implement Output Generation**:
   - Create consistent CSV output generation
   - Develop standardized plotting functions
   - Add summary report generation

## Quality Guidelines

1. **Code Organization**:
   - Keep files small and focused (< 300 lines per file)
   - Separate concerns clearly
   - Use descriptive naming

2. **Documentation**:
   - Document all functions and classes
   - Include example usage
   - Document mathematical formulas and statistical techniques

3. **Error Handling**:
   - Provide meaningful error messages
   - Log all data processing steps
   - Implement graceful failure modes

4. **Testing**:
   - Add unit tests for core functions
   - Include test data for reproducibility
   - Add validation of model outputs

## Delivery

The final deliverable should include:

1. Fully functional codebase implementing all requirements
2. Documentation of each model's approach and limitations
3. Sample outputs demonstrating forecasting accuracy
4. Instructions for running the application
