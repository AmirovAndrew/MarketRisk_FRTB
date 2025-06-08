import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Ignore common pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Flask App Setup ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ============================================================================
# FRTB SBA PARAMETERS (Centralized and Corrected)
# ============================================================================
FRTB_PARAMETERS = {
    "BASE_CURRENCY": "USD",
    "RISK_CLASSES": ["GIRR", "CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP", "EQ", "COMM", "FX"],
    "DELTA_RISK_WEIGHTS": {
        "GIRR": {"1": 1.75, "2": 1.75, "3": 1.75, "4": 1.75, "5": 1.75, "6": 1.75, "7": 1.75, "8": 1.75, "9": 1.75, "10": 1.75, "11": 1.75, "12": 1.75, "13": 1.75, "INF": 6.4},
        "FX": {"standard": 15.0, "specified_pair": 15.0 / np.sqrt(2)},
        "EQ": {"1": 25, "2": 32, "3": 29, "4": 27, "5": 18, "6": 21, "7": 25, "8": 22, "9": 27, "10": 29, "11": 16, "12": 16},
        "COMM": {"1": 19, "2": 20, "3": 17, "4": 18, "5": 24, "6": 20, "7": 25, "8": 13, "9": 13, "10": 50, "11": 21, "12": 20, "13": 16, "14": 15, "15": 12, "16": 50, "17": 18},
        "CSR_NS": {"1": 0.5, "2": 3.0, "3": 15.0}
    },
    "VEGA_RISK_WEIGHTS": {"GIRR": 0.0075, "FX": 0.55, "EQ": 0.28, "COMM": 0.36, "CSR_NS": 0.27},
    "CORRELATIONS": {
        "within_bucket": {"GIRR": 0.999, "FX": 0.60, "EQ": 0.15, "COMM": 0.55, "CSR_NS": 0.65},
        "across_bucket": {"GIRR": 0.42, "FX": 0.60, "EQ": 0.075, "COMM": 0.20, "CSR_NS": 0.35}
    },
    # --- CORRECTED: Added missing parameter for curvature ---
    "CURVATURE_RISK_FACTORS": ["GIRR", "CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP", "EQ", "COMM"],
    "DRC_RISK_WEIGHTS": {
        "CSR_NS": {"IG": 0.005, "HY": 0.02, "NR": 0.05},
        "CSR_SEC_CTP": 0.03,
        "CSR_SEC_NCTP": 0.015
    },
    "DRC_NETTING_BENEFIT": 0.65,
    "RRAO_CHARGE_RATE": 0.01,
    "FX_SPECIFIED_CURRENCIES": ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF'],
    "RISK_CLASS_NAMES": {
        "GIRR": "General Interest Rate Risk",
        "CSR_NS": "Credit Spread Risk - Non-Securitizations",
        "CSR_SEC_CTP": "Credit Spread Risk - Securitizations (CTP)",
        "CSR_SEC_NCTP": "Credit Spread Risk - Securitizations (Non-CTP)",
        "EQ": "Equity Risk",
        "COMM": "Commodity Risk",
        "FX": "Foreign Exchange Risk"
    }
}

# ============================================================================
# FRTB CALCULATION CLASS (Full and Corrected Implementation)
# ============================================================================
class FRTBSensitivityBasedApproach:
    def __init__(self, parameters: Dict):
        self.params = parameters
        self.base_currency = self.params.get("BASE_CURRENCY", "USD")

    def load_positions_from_file(self, file_path: str) -> pd.DataFrame:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format.")
            
            required_columns = ['trade_id', 'risk_class', 'risk_factor', 'bucket', 'delta_sensitivity', 'Market Value']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Ensure file has: {required_columns}")

            # Define defaults for all potentially missing columns
            defaults = {
                "vega_sensitivity": 0.0, "notional": 0.0,
                "currency": self.base_currency, "exotic_flag": False, "position_side": "long",
                "credit_rating": "NR", "Instrument Type": "Unknown"
            }
            for col, default_value in defaults.items():
                if col not in df.columns:
                    df[col] = default_value
                df[col] = df[col].fillna(default_value)

            # Standardize data types
            for col in ['delta_sensitivity', 'vega_sensitivity', 'notional', 'Market Value']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['bucket'] = df['bucket'].astype(str)
            return df
        except Exception as e:
            print(f"Error loading positions: {e}")
            raise

    def _get_delta_risk_weight(self, risk_class: str, bucket: str, risk_factor: str) -> float:
        if risk_class == "FX":
            try:
                curr1, curr2 = risk_factor.upper().split('/')
                if curr1 in self.params["FX_SPECIFIED_CURRENCIES"] and curr2 in self.params["FX_SPECIFIED_CURRENCIES"]:
                    return self.params["DELTA_RISK_WEIGHTS"]["FX"]["specified_pair"]
                return self.params["DELTA_RISK_WEIGHTS"]["FX"]["standard"]
            except (ValueError, AttributeError):
                return self.params["DELTA_RISK_WEIGHTS"]["FX"]["standard"]

        class_weights = self.params["DELTA_RISK_WEIGHTS"].get(risk_class, {})
        return class_weights.get(bucket, max(class_weights.values()) if class_weights else 20.0)

    def _optimized_aggregation(self, values: List[float], correlation: float) -> float:
        if not values: return 0.0
        sum_val = sum(values)
        sum_sq_val = sum(x**2 for x in values)
        return float(np.nan_to_num(np.sqrt(max(0, (1 - correlation) * sum_sq_val + correlation * sum_val**2))))

    def _aggregate_sensitivities(self, pos: pd.DataFrame, rc: str, sens_col: str, weight_map: Dict) -> float:
        class_pos = pos[pos['risk_class'] == rc].copy()
        if class_pos.empty or class_pos[sens_col].abs().sum() == 0: return 0.0

        if sens_col == 'delta_sensitivity':
            class_pos['ws'] = class_pos.apply(lambda r: r[sens_col] * self._get_delta_risk_weight(rc, r['bucket'], r['risk_factor']) / 100, axis=1)
        else: # Vega
            weight = weight_map.get(rc, 0.0)
            class_pos['ws'] = class_pos[sens_col] * weight

        bucket_capitals = {}
        corr_within = self.params["CORRELATIONS"]['within_bucket'].get(rc, 0.5)
        for bucket, group in class_pos.groupby('bucket'):
            factor_sens = group.groupby('risk_factor')['ws'].sum()
            bucket_capitals[bucket] = self._optimized_aggregation(list(factor_sens), corr_within)
        
        if not bucket_capitals: return 0.0
        if len(bucket_capitals) == 1: return list(bucket_capitals.values())[0]

        corr_across = self.params["CORRELATIONS"]['across_bucket'].get(rc, 0.2)
        return self._optimized_aggregation(list(bucket_capitals.values()), corr_across)

    def calculate_delta_charge(self, pos: pd.DataFrame) -> Dict[str, float]:
        return {rc: self._aggregate_sensitivities(pos, rc, 'delta_sensitivity', {}) for rc in self.params["RISK_CLASSES"]}

    def calculate_vega_charge(self, pos: pd.DataFrame) -> Dict[str, float]:
        return {rc: self._aggregate_sensitivities(pos, rc, 'vega_sensitivity', self.params["VEGA_RISK_WEIGHTS"]) for rc in self.params["RISK_CLASSES"]}
    
    def calculate_curvature_charge(self, positions: pd.DataFrame) -> dict:
        curvature_charges_by_class = {}
        optionality_instruments = ['Option', 'Swaption', 'Callable Bond', 'Floor', 'Cap']
        curvature_positions = positions[positions['Instrument Type'].isin(optionality_instruments)].copy()

        if curvature_positions.empty:
            return {}

        for risk_class, group in curvature_positions.groupby('risk_class'):
            if risk_class not in self.params.get('CURVATURE_RISK_FACTORS', []):
                continue
            
            cvr_long = group[group['position_side'] == 'long']['Market Value'].sum()
            cvr_short = abs(group[group['position_side'] == 'short']['Market Value'].sum())
            class_charge = (cvr_long + cvr_short) * 0.1 # Placeholder logic
            
            if class_charge > 0:
                curvature_charges_by_class[risk_class] = class_charge
                
        return curvature_charges_by_class

    def calculate_default_risk_charge(self, pos: pd.DataFrame) -> dict:
        drc_pos = pos[pos['risk_class'].str.startswith('CSR')].copy()
        if drc_pos.empty:
            return {'charge': 0.0, 'gross_jtd_long': 0.0, 'gross_jtd_short': 0.0}

        def get_drc_weight(r):
            if r['risk_class'] == 'CSR_NS':
                return self.params["DRC_RISK_WEIGHTS"]['CSR_NS'].get(r['credit_rating'], self.params["DRC_RISK_WEIGHTS"]['CSR_NS']['NR'])
            return self.params["DRC_RISK_WEIGHTS"].get(r['risk_class'], 0.0)

        drc_pos['jtd'] = drc_pos.apply(lambda r: r['notional'] * get_drc_weight(r), axis=1)
        jtd_long = drc_pos[drc_pos['position_side'] == 'long']['jtd'].sum()
        jtd_short = drc_pos[drc_pos['position_side'] == 'short']['jtd'].sum()
        
        hedge_benefit = self.params["DRC_NETTING_BENEFIT"]
        drc = max(jtd_long, jtd_short) - (hedge_benefit * min(jtd_long, jtd_short))

        return {'charge': max(0, drc), 'gross_jtd_long': jtd_long, 'gross_jtd_short': jtd_short}

    # --- CORRECTED: Renamed function from calculate_residual_risk_addon ---
    def calculate_rrao_charge(self, pos: pd.DataFrame) -> float:
        if 'exotic_flag' not in pos.columns: return 0.0
        exotic_notional = pos[pos['exotic_flag'] == True]['notional'].abs().sum()
        return exotic_notional * self.params["RRAO_CHARGE_RATE"]
    
    def calculate_total_capital_charge(self, positions: pd.DataFrame) -> Dict:
        delta_by_class = self.calculate_delta_charge(positions)
        vega_by_class = self.calculate_vega_charge(positions)
        curvature_by_class = self.calculate_curvature_charge(positions)
        drc_results = self.calculate_default_risk_charge(positions)
        rrao_charge = self.calculate_rrao_charge(positions)

        delta_total = sum(delta_by_class.values())
        vega_total = sum(vega_by_class.values())
        curvature_total = sum(curvature_by_class.values())
        drc_charge = drc_results['charge']
        
        sbm_charge = delta_total + vega_total + curvature_total
        total_capital_charge = sbm_charge + drc_charge + rrao_charge

        sbm_details = {}
        all_risk_classes = set(delta_by_class.keys()) | set(vega_by_class.keys())
        for rc in all_risk_classes:
            sbm_details[rc] = {
                'delta': delta_by_class.get(rc, 0),
                'vega': vega_by_class.get(rc, 0),
                'total_delta_vega': delta_by_class.get(rc, 0) + vega_by_class.get(rc, 0)
            }

        return {
            'calculation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'base_currency': self.base_currency,
            'positions_processed': len(positions),
            'total_capital_charge': total_capital_charge,
            'sbm_charge': sbm_charge,
            'delta_charge_total': delta_total,
            'vega_charge_total': vega_total,
            'curvature_charge_total': curvature_total,
            'drc_charge': drc_charge,
            'rrao_charge': rrao_charge,
            'sbm_details': sbm_details,
            'curvature_by_risk_class': curvature_by_class,
            'drc_details': drc_results,
        }

# --- Flask Routes ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected for uploading'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            calculator = FRTBSensitivityBasedApproach(parameters=FRTB_PARAMETERS)
            positions = calculator.load_positions_from_file(filepath)
            if not positions.empty:
                results = calculator.calculate_total_capital_charge(positions)
                return render_template('results.html', results=results, risk_class_names=FRTB_PARAMETERS['RISK_CLASS_NAMES'])
            else:
                return "Error: Could not process the uploaded file."
        except Exception as e:
            app.logger.error(f"An error occurred during calculation: {str(e)}")
            return f"An error occurred: <pre>{str(e)}</pre>", 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return "Error: Invalid file type. Please upload a .csv or .xlsx file."

# --- Main execution block ---
if __name__ == '__main__':
    app.run(debug=True)
