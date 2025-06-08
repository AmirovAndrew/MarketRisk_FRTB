# FRTB Market Risk Calculator

## 1. Project Description

This project is a web-based application designed to calculate market risk capital requirements according to the **Fundamental Review of the Trading Book (FRTB)** framework, specifically using the **Sensitivities-Based Approach (SBA)**.

Built with Python and the Flask web framework, the application provides a user-friendly interface to upload a portfolio of financial positions in `.csv` or `.xlsx` format. It then processes the data and generates a detailed, interactive risk report complete with dynamic charts and data tables, breaking down the capital charge into its core components.

The calculation engine is designed to be modular, with all regulatory parameters (risk weights, correlations, etc.) centralized for easy auditing and updates.

## 2. Key Features

* **FRTB-SBA Engine:** Calculates capital charges for the five key components:
    * Delta Risk
    * Vega Risk
    * Curvature Risk
    * Default Risk Charge (DRC)
    * Residual Risk Add-On (RRAO)
* **Interactive Web Interface:** A clean and modern UI allows users to easily upload files and view results.
* **Dynamic Reporting:** Generates a professional risk report with data tables and visualizations powered by Chart.js.
* **Flexible File Support:** Accepts portfolio data in both **CSV (`.csv`)** and **Excel (`.xlsx`, `.xls`)** formats.
* **Robust Data Handling:** The application gracefully handles files with missing optional columns by applying default values.
* **Centralized Parameters:** All FRTB risk weights and parameters are stored in a single Python dictionary, making them easy to configure.


## 3. Installation

To get the application running on your local machine, follow these steps.

### Prerequisites

* Python 3.8 or newer
* `pip` (Python package installer)


### Step 1: Set Up the Project

First, create a project directory and navigate into it.

```bash
mkdir frtb-calculator
cd frtb-calculator
```

Place the main Python script (e.g., `app.py`) inside this directory. Create the required sub-directories for templates and static files:

```bash
mkdir templates
mkdir static
```

Your final project structure should look like this:

```
/frtb-calculator
├── app.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   ├── results.html
│   └── charts.html
└── uploads/          <-- This will be created automatically
```


### Step 2: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```


### Step 3: Install Dependencies

Create a file named `requirements.txt` in your project root with the following content:

**`requirements.txt`**:

```
Flask
pandas
numpy
openpyxl
```

Now, install these packages using `pip`:

```bash
pip install -r requirements.txt
```


## 4. Running the Application

With the dependencies installed, you can run the application using the Flask command-line interface.

```bash
# Ensure you are in the root project directory (frtb-calculator)
flask --app app run --debug
```

You will see output indicating that the development server is running, typically on `http://127.0.0.1:5000/`.

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Open a web browser and navigate to **http://127.0.0.1:5000/** to use the application.

## 5. How to Use

1. **Navigate to the Homepage:** Open your browser to the root URL.
2. **Upload File:** Click the "Choose a File..." button and select your positions file (`.csv` or `.xlsx`). The name of the selected file will appear.
3. **Calculate Risk:** Click the "Calculate Risk" button.
4. **View Report:** You will be redirected to the results page, which displays the full breakdown of the market risk capital charge.

## 6. Input File Format

Your input file must contain a header row with the following column names. The column order does not matter.

### Required Columns

These columns must be present in your file.


| Column Name | Type | Description |
| :-- | :-- | :-- |
| `trade_id` | String | A unique identifier for the position. |
| `risk_class` | String | The FRTB risk class (e.g., `GIRR`, `CSR_NS`, `EQ`, `FX`, `COMM`). |
| `risk_factor` | String | The specific risk factor (e.g., `USD_GOVT_10Y`, `AAPL`, `EUR/USD`). |
| `bucket` | String | The regulatory bucket for the risk factor. |
| `delta_sensitivity` | Number | The first-order sensitivity (Delta) of the position. |
| `Market Value` | Number | The current market value of the position in the base currency. |

### Optional Columns

If these columns are not present, the application will use default values.


| Column Name | Type | Description |
| :-- | :-- | :-- |
| `vega_sensitivity` | Number | The vega sensitivity, required for positions with optionality. Default: `0`. |
| `notional` | Number | The notional amount of the position, used for DRC and RRAO. Default: `0`. |
| `exotic_flag` | Boolean | `True` if the position is an exotic, for the RRAO calculation. Default: `False`. |
| `position_side` | String | `long` or `short`. Used for DRC and Curvature. Default: `long`. |
| `credit_rating` | String | The credit rating (e.g., `IG`, `HY`, `NR`), used for DRC. Default: `NR`. |
| `Instrument Type` | String | The type of instrument (e.g., `Option`, `Bond`, `Swaption`), used for Curvature. Default: `Unknown`. |
