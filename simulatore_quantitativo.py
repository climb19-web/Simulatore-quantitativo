import datetime
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================================
# SIMULATORE QUANTITATIVO - FOOTPRINT, CARBONIO, CLIMA
# ============================================================================

st.set_page_config(
    page_title="Simulatore Quantitativo Eco-Evolutivo",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("SIMULATORE QUANTITATIVO ECO-EVOLUTIVO 2075")
st.caption(
    "Modello data-informed: biocapacita, impronta ecologica, emissioni CO2, ppm e temperatura."
)


# ============================================================================
# PARAMETRI DI MODELLO
# ============================================================================

CSV_PATH = Path("export_custom_range.csv")
START_YEAR = 2026
END_YEAR = 2075

# Valori climatici di riferimento. Le emissioni sono CO2, non CO2e.
CARBON_PARAMS = {
    "baseline_emissions_gtco2": 40.6,  # GtCO2/anno, ordine di grandezza GCB 2023 antropogenico totale
    "baseline_ppm": 425.0,
    "baseline_temp": 1.30,  # aumento rispetto al preindustriale al punto di partenza
    "ppm_per_gtco2": 7.82,  # GtCO2 per 1 ppm
    "airborne_fraction": 0.47,  # quota media che resta in atmosfera
    "ecs": 3.0,  # sensibilita climatica al raddoppio della CO2
}

PRACTICAL_BASELINES = {
    "meat_kg_per_person_year": 43.0,
}

ENERGY_BASELINES = {
    "Mondo": {
        "Petrolio": 31.0,
        "Carbone": 27.0,
        "Gas naturale": 23.5,
        "Nucleare": 4.0,
        "Idroelettrico": 6.0,
        "Rinnovabili moderne": 6.5,
        "Bioenergie e altro": 2.0,
    },
    "Europa": {
        "Petrolio": 34.0,
        "Carbone": 11.0,
        "Gas naturale": 23.0,
        "Nucleare": 10.0,
        "Idroelettrico": 6.0,
        "Rinnovabili moderne": 13.0,
        "Bioenergie e altro": 3.0,
    },
}

FOSSIL_ENERGY_SOURCES = ["Petrolio", "Carbone", "Gas naturale"]
LOW_CARBON_ENERGY_SOURCES = [
    "Nucleare",
    "Idroelettrico",
    "Rinnovabili moderne",
    "Bioenergie e altro",
]

# Proiezione ONU semplificata, usata per modulare gli scenari demografici.
UN_POP_YEARS = np.array([2026, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075])
UN_POP_VALUES = np.array([8.30e9, 8.69e9, 9.03e9, 9.35e9, 9.63e9, 9.88e9, 10.07e9, 10.21e9, 10.31e9, 10.38e9, 10.41e9])

EF_COMPONENTS = [
    "EF Carbon",
    "EF Cropland",
    "EF Grazing Land",
    "EF Forest Products",
    "EF Fishing Grounds",
    "EF Built-up Land",
]

BIOCAP_COMPONENTS = [
    "Biocap Built-up Land",
    "Biocap Cropland",
    "Biocap Fishing Grounds",
    "Biocap Forest Products",
    "Biocap Grazing Land",
]

EF_LABELS_IT = {
    "EF Carbon": "Carbonio",
    "EF Cropland": "Coltivazioni",
    "EF Grazing Land": "Pascolo",
    "EF Forest Products": "Prodotti forestali",
    "EF Fishing Grounds": "Pesca",
    "EF Built-up Land": "Suolo costruito",
}


# ============================================================================
# CARICAMENTO E CALIBRAZIONE DATI FOOTPRINT
# ============================================================================

if "area_dati" not in st.session_state:
    st.session_state.area_dati = "Mondo"
if "tipo_scenario_europa" not in st.session_state:
    st.session_state.tipo_scenario_europa = "Trend recente"


@st.cache_data
def load_footprint_csv(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"File CSV non trovato: {filepath}")

    df = pd.read_csv(filepath)
    required = {"Anno", "Paese", "BiocapTotGHA", "EFConsTotGHA", "Numero di Terre consumate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {', '.join(sorted(missing))}")

    numeric_cols = [
        "Anno",
        "BiocapPerCap",
        "EFConsPerCap",
        "BiocapTotGHA",
        "EFConsTotGHA",
        "Numero di Terre consumate",
        "Popolazione",
        *EF_COMPONENTS,
        *BIOCAP_COMPONENTS,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Anno", "BiocapTotGHA", "EFConsTotGHA"])
    df["Anno"] = df["Anno"].astype(int)
    return df.sort_values(["Paese", "Anno"]).reset_index(drop=True)


def select_area_footprint(df: pd.DataFrame, area: str) -> pd.DataFrame:
    area_df = df[df["Paese"].astype(str).str.lower().eq(area.lower())].copy()
    if area_df.empty:
        raise ValueError(f"Il CSV non contiene righe con Paese = {area}.")

    quality_col = get_quality_column(area_df)
    if quality_col:
        area_df["_quality_rank"] = area_df[quality_col].astype(str).str.upper().eq("PROIEZIONE").astype(int)
    else:
        area_df["_quality_rank"] = 0

    area_df = (
        area_df.sort_values(["Anno", "_quality_rank"])
        .groupby("Anno", as_index=False)
        .first()
        .drop(columns=["_quality_rank"], errors="ignore")
    )
    area_df["Rapporto Area"] = area_df["EFConsTotGHA"] / area_df["BiocapTotGHA"]
    return area_df.sort_values("Anno").reset_index(drop=True)


def get_quality_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if col.lower().startswith("punteggio")), None)


def cagr(start_value: float, end_value: float, years: int) -> float:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def compute_recent_trends(world: pd.DataFrame, start_year: int = 2018, end_year: int = 2026) -> dict:
    start = world[world["Anno"].eq(start_year)]
    end = world[world["Anno"].eq(end_year)]
    if start.empty or end.empty:
        return {"ef": 0.004, "biocap": 0.002}

    years = end_year - start_year
    return {
        "ef": cagr(float(start.iloc[0]["EFConsTotGHA"]), float(end.iloc[0]["EFConsTotGHA"]), years),
        "biocap": cagr(float(start.iloc[0]["BiocapTotGHA"]), float(end.iloc[0]["BiocapTotGHA"]), years),
    }


def select_baseline(world: pd.DataFrame, year: int = START_YEAR) -> pd.Series:
    row = world[world["Anno"].eq(year)]
    if row.empty:
        return world.iloc[-1]
    return row.iloc[0]


try:
    footprint_data = load_footprint_csv(CSV_PATH)
    available_areas = [area for area in ["Mondo", "Europa"] if area in set(footprint_data["Paese"].astype(str))]
    if not available_areas:
        available_areas = sorted(footprint_data["Paese"].dropna().astype(str).unique().tolist())

    selected_area = st.session_state.get("area_dati", "Mondo")
    if selected_area not in available_areas:
        selected_area = "Mondo" if "Mondo" in available_areas else available_areas[0]
        st.session_state.area_dati = selected_area

    world_reference_data = select_area_footprint(footprint_data, "Mondo")
    world_reference_baseline = select_baseline(world_reference_data)
    world_reference_carbon_ef_gha = float(world_reference_baseline.get("EF Carbon", world_reference_baseline["EFConsTotGHA"])) / 1e9

    world_data = select_area_footprint(footprint_data, selected_area)
    baseline = select_baseline(world_data)
    recent_trends = compute_recent_trends(world_data)
    data_error = None
except Exception as exc:
    footprint_data = pd.DataFrame()
    available_areas = ["Mondo", "Europa"]
    selected_area = st.session_state.get("area_dati", "Mondo")
    world_reference_carbon_ef_gha = 13.262
    world_data = pd.DataFrame()
    baseline = pd.Series(
        {
            "Anno": START_YEAR,
            "BiocapTotGHA": 12.144e9,
            "EFConsTotGHA": 21.599e9,
            "Numero di Terre consumate": 1.783,
            "Popolazione": 8.313e9,
            "EF Carbon": 13.262e9,
            "EF Cropland": 3.808e9,
            "EF Grazing Land": 1.008e9,
            "EF Forest Products": 2.272e9,
            "EF Fishing Grounds": 0.696e9,
            "EF Built-up Land": 0.554e9,
        }
    )
    recent_trends = {"ef": 0.004, "biocap": 0.002}
    data_error = str(exc)

def scenario_default_trends(trends: dict, scenario_type: str) -> tuple[float, float]:
    if scenario_type == "Consumo costante":
        return 0.0, round(trends["biocap"] * 100, 2)
    return round(trends["ef"] * 100, 2), round(trends["biocap"] * 100, 2)


effective_scenario_type = (
    st.session_state.tipo_scenario_europa if selected_area.lower() == "europa" else "Trend recente"
)


if (
    st.session_state.get("_area_dati_attiva") != selected_area
    or st.session_state.get("_tipo_scenario_base_attivo") != effective_scenario_type
):
    default_ef, default_biocap = scenario_default_trends(
        recent_trends, effective_scenario_type
    )
    st.session_state.trend_ef = default_ef
    st.session_state.trend_biocap = default_biocap
    st.session_state._area_dati_attiva = selected_area
    st.session_state._tipo_scenario_base_attivo = effective_scenario_type


# ============================================================================
# SESSION STATE
# ============================================================================

initial_states = {
    "trend_ef": round(recent_trends["ef"] * 100, 2),
    "trend_biocap": round(recent_trends["biocap"] * 100, 2),
    "leva_carbonio": 0.0,
    "leva_dieta": 0.0,
    "leva_forestazione": 0.0,
    "rimozione_co2": 0.0,
    "spinta_rinnovabili": 0.0,
    "spinta_nucleare": 0.0,
    "efficienza_energia": 0.0,
    "deviazione_pop": 0.0,
    "danno_clima": 0.0,
}

for key, value in initial_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


def apply_bau():
    default_ef, default_biocap = scenario_default_trends(
        recent_trends, effective_scenario_type
    )
    st.session_state.trend_ef = default_ef
    st.session_state.trend_biocap = default_biocap
    st.session_state.leva_carbonio = 0.0
    st.session_state.leva_dieta = 0.0
    st.session_state.leva_forestazione = 0.0
    st.session_state.rimozione_co2 = 0.0
    st.session_state.spinta_rinnovabili = 0.0
    st.session_state.spinta_nucleare = 0.0
    st.session_state.efficienza_energia = 0.0
    st.session_state.deviazione_pop = 0.0
    st.session_state.danno_clima = 0.0


def apply_transizione_moderata():
    apply_bau()
    st.session_state.leva_carbonio = 40.0
    st.session_state.leva_dieta = 20.0
    st.session_state.leva_forestazione = 0.20
    st.session_state.rimozione_co2 = 2.0
    st.session_state.spinta_rinnovabili = 25.0
    st.session_state.spinta_nucleare = 5.0
    st.session_state.efficienza_energia = 15.0


def apply_transizione_forte():
    apply_bau()
    st.session_state.leva_carbonio = 70.0
    st.session_state.leva_dieta = 35.0
    st.session_state.leva_forestazione = 0.40
    st.session_state.rimozione_co2 = 5.0
    st.session_state.spinta_rinnovabili = 45.0
    st.session_state.spinta_nucleare = 10.0
    st.session_state.efficienza_energia = 25.0


def apply_stress_ecologico():
    apply_bau()
    st.session_state.trend_ef = 0.80
    st.session_state.trend_biocap = -0.30
    st.session_state.danno_clima = 0.20


# ============================================================================
# MODELLO
# ============================================================================

def temp_from_ppm(ppm: float) -> float:
    if ppm <= 0:
        return CARBON_PARAMS["baseline_temp"]
    delta_from_baseline = CARBON_PARAMS["ecs"] * math.log(
        ppm / CARBON_PARAMS["baseline_ppm"], 2
    )
    return CARBON_PARAMS["baseline_temp"] + delta_from_baseline


def overshoot_day(year: int, ratio_earths: float) -> str:
    if ratio_earths <= 1:
        return "Sostenibile"
    available_days = max(1, min(365, int(365 / ratio_earths)))
    day = datetime.date(year, 1, 1) + datetime.timedelta(days=available_days - 1)
    months = {
        "January": "Gennaio",
        "February": "Febbraio",
        "March": "Marzo",
        "April": "Aprile",
        "May": "Maggio",
        "June": "Giugno",
        "July": "Luglio",
        "August": "Agosto",
        "September": "Settembre",
        "October": "Ottobre",
        "November": "Novembre",
        "December": "Dicembre",
    }
    day_str, month_en = day.strftime("%d %B").split()
    return f"{day_str} {months[month_en]}"


def practical_interpretation(params: dict, baseline_row: pd.Series) -> dict:
    years = END_YEAR - START_YEAR
    meat_base = PRACTICAL_BASELINES["meat_kg_per_person_year"]
    meat_target = meat_base * (1 - params["leva_dieta"] / 100)
    meat_reduction = meat_base - meat_target

    base_biocap_gha = float(baseline_row["BiocapTotGHA"]) / 1e9
    restoration_factor = (1 + params["leva_forestazione"] / 100) ** years
    extra_biocap_gha = base_biocap_gha * (restoration_factor - 1)
    extra_biocap_km2_million = extra_biocap_gha * 10

    cumulative_removal = params["rimozione_co2"] * years / 2
    energy_baseline = get_energy_baseline(selected_area)
    energy_final = energy_mix_for_progress(params, 1.0, selected_area)
    fossil_base = energy_fossil_share(energy_baseline)
    fossil_final = energy_fossil_share(energy_final)
    fossil_share_factor = fossil_final / fossil_base if fossil_base > 0 else 1.0
    explicit_carbon_factor = 1 - params["leva_carbonio"] / 100
    efficiency_factor = 1 - params["efficienza_energia"] / 100
    effective_carbon_factor = max(0.0, explicit_carbon_factor * fossil_share_factor * efficiency_factor)
    effective_carbon_reduction = (1 - effective_carbon_factor) * 100
    energy_only_reduction = (1 - max(0.0, fossil_share_factor * efficiency_factor)) * 100
    fossil_points_reduction = fossil_base - fossil_final

    return {
        "meat_target": meat_target,
        "meat_reduction": meat_reduction,
        "carbon_reduction": params["leva_carbonio"],
        "effective_carbon_reduction": effective_carbon_reduction,
        "energy_only_reduction": energy_only_reduction,
        "fossil_points_reduction": fossil_points_reduction,
        "food_reduction": params["leva_dieta"],
        "extra_biocap_gha": extra_biocap_gha,
        "extra_biocap_km2_million": extra_biocap_km2_million,
        "co2_removal_2075": params["rimozione_co2"],
        "co2_removal_cumulative": cumulative_removal,
    }


def get_energy_baseline(area: str) -> dict:
    return ENERGY_BASELINES.get(area, ENERGY_BASELINES["Mondo"])


def energy_mix_for_progress(params: dict, progress: float, area: str) -> dict:
    mix = get_energy_baseline(area).copy()
    renewable_add = params["spinta_rinnovabili"] * progress
    nuclear_add = params["spinta_nucleare"] * progress
    total_shift = min(sum(mix[src] for src in FOSSIL_ENERGY_SOURCES), renewable_add + nuclear_add)

    requested_shift = renewable_add + nuclear_add
    if requested_shift > 0:
        renewable_add = total_shift * renewable_add / requested_shift
        nuclear_add = total_shift * nuclear_add / requested_shift

    fossil_total = sum(mix[src] for src in FOSSIL_ENERGY_SOURCES)
    if fossil_total > 0:
        for source in FOSSIL_ENERGY_SOURCES:
            mix[source] -= total_shift * mix[source] / fossil_total

    mix["Rinnovabili moderne"] += renewable_add
    mix["Nucleare"] += nuclear_add

    # Correzione numerica per mantenere la somma a 100.
    total = sum(mix.values())
    if total > 0:
        mix = {source: value * 100 / total for source, value in mix.items()}
    return mix


def energy_fossil_share(mix: dict) -> float:
    return sum(mix[src] for src in FOSSIL_ENERGY_SOURCES)


def get_configured_gemini_key(manual_key: str = "") -> str:
    if manual_key.strip():
        return manual_key.strip()

    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            secret_value = st.secrets.get(key_name)
        except Exception:
            secret_value = None
        if secret_value:
            return str(secret_value)

        env_value = os.getenv(key_name)
        if env_value:
            return env_value

    return ""


def build_ai_context() -> str:
    context_ratio_unit = "Terre" if selected_area.lower() == "mondo" else "Europe"
    return f"""
Scenario corrente del simulatore:
- Area dati: {selected_area}
- Tipo scenario base: {effective_scenario_type}
- Trend impronta: {params['trend_ef']:.2f}% annuo
- Trend biocapacita: {params['trend_biocap']:.2f}% annuo
- Riduzione impronta carbonio al 2075: {params['leva_carbonio']:.1f}%
- Riduzione impronta agro-alimentare al 2075: {params['leva_dieta']:.1f}%
- Extra crescita biocapacita: {params['leva_forestazione']:.2f}% annuo
- Rimozione CO2 al 2075: {params['rimozione_co2']:.1f} GtCO2/anno
- Aumento rinnovabili moderne al 2075: {params['spinta_rinnovabili']:.1f} punti %
- Aumento nucleare al 2075: {params['spinta_nucleare']:.1f} punti %
- Efficienza energia al 2075: {params['efficienza_energia']:.1f}%

Risultati 2075:
- Impronta: {last['Impronta Totale (Gha)']:.2f} Gha
- Biocapacita: {last['Biocapacita (Gha)']:.2f} Gha
- Rapporto: {last['Rapporto (Terre)']:.2f} {context_ratio_unit}
- Emissioni nette: {last['Emissioni Nette (GtCO2/anno)']:.1f} GtCO2/anno
- CO2 atmosferica: {last['CO2 (ppm)']:.0f} ppm
- Temperatura: +{last['Temperatura (C)']:.2f} C
- Overshoot Day: {last['Overshoot Day']}
"""


def ask_gemini(question: str, api_key: str, model_name: str) -> str:
    prompt = f"""
Sei un assistente IA generale integrato in un simulatore eco-evolutivo.
Rispondi in italiano, in modo chiaro, utile e proporzionato alla domanda.

Regole:
- Puoi rispondere anche a domande generali, non limitate al simulatore.
- Se la domanda riguarda lo scenario corrente, usa il contesto del modello qui sotto.
- Se la domanda richiede conoscenze esterne, rispondi come assistente generale e segnala quando stai usando conoscenza generale o stime.
- Distingui quando opportuno tra dati del simulatore, dati ufficiali citati dal modello, stime e spiegazioni generali.
- Non modificare i risultati del modello: puoi interpretarli, confrontarli e contestualizzarli.

Contesto dello scenario corrente, da usare solo se rilevante:
{build_ai_context()}

Domanda dell'utente:
{question}
"""
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "API_KEY_INVALID" in detail:
            raise RuntimeError(
                "API key Gemini non valida. Controlla di aver copiato la chiave completa da Google AI Studio, "
                "senza spazi o virgolette, e che la chiave sia abilitata per Gemini/Generative Language API."
            ) from exc
        raise RuntimeError(f"Errore Gemini HTTP {exc.code}: {detail}") from exc

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Nessuna risposta Gemini ricevuta: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Risposta Gemini senza testo: {data}")
    return "\n".join(text_parts)


def component_value(row: pd.Series, col: str) -> float:
    value = row.get(col, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value) / 1e9


def run_model(params: dict) -> pd.DataFrame:
    years = list(range(START_YEAR, END_YEAR + 1))
    total_steps = END_YEAR - START_YEAR

    base_ef_total = float(baseline["EFConsTotGHA"]) / 1e9
    base_biocap_total = float(baseline["BiocapTotGHA"]) / 1e9
    base_population = float(baseline.get("Popolazione", UN_POP_VALUES[0]))
    if pd.isna(base_population) or base_population <= 0:
        biocap_per_cap = float(baseline.get("BiocapPerCap", 0) or 0)
        ef_per_cap = float(baseline.get("EFConsPerCap", 0) or 0)
        if biocap_per_cap > 0:
            base_population = float(baseline["BiocapTotGHA"]) / biocap_per_cap
        elif ef_per_cap > 0:
            base_population = float(baseline["EFConsTotGHA"]) / ef_per_cap
        else:
            base_population = UN_POP_VALUES[0]

    base_components = {col: component_value(baseline, col) for col in EF_COMPONENTS}
    component_sum = sum(base_components.values())
    if component_sum <= 0:
        base_components = {"EF Carbon": base_ef_total}
        component_sum = base_ef_total

    # Normalizza eventuali differenze di arrotondamento rispetto al totale ufficiale.
    scale = base_ef_total / component_sum
    base_components = {col: value * scale for col, value in base_components.items()}
    base_carbon_ef = max(base_components.get("EF Carbon", 0.0), 1e-9)

    cumulative_atmospheric_gtco2 = 0.0
    cumulative_atmospheric_without_removal_gtco2 = 0.0
    cumulative_gross_emissions = 0.0
    cumulative_removed = 0.0
    rows = []

    for year in years:
        step = year - START_YEAR
        progress = step / total_steps if total_steps else 0

        un_population = np.interp(year, UN_POP_YEARS, UN_POP_VALUES)
        un_growth_factor = un_population / UN_POP_VALUES[0]
        reference_population = base_population * un_growth_factor
        population = reference_population * ((1 + params["deviazione_pop"] / 100) ** step)
        # Il trend impronta e gia calcolato sull'impronta totale osservata/proiettata.
        # La popolazione modifica l'impronta solo se l'utente imposta una deviazione dal trend base.
        population_factor = (1 + params["deviazione_pop"] / 100) ** step
        energy_baseline = get_energy_baseline(selected_area)
        energy_mix = energy_mix_for_progress(params, progress, selected_area)
        fossil_share_factor = energy_fossil_share(energy_mix) / energy_fossil_share(energy_baseline)
        efficiency_factor = max(0.0, 1 - (params["efficienza_energia"] / 100) * progress)

        ef_growth = (1 + params["trend_ef"] / 100) ** step
        biocap_growth = (1 + (params["trend_biocap"] + params["leva_forestazione"]) / 100) ** step

        carbon_reduction = 1 - (params["leva_carbonio"] / 100) * progress
        food_reduction = 1 - (params["leva_dieta"] / 100) * progress
        carbon_reduction = max(0.0, carbon_reduction)
        food_reduction = max(0.0, food_reduction)

        ef_parts = {}
        for col, value in base_components.items():
            modifier = ef_growth * population_factor
            if col == "EF Carbon":
                modifier *= carbon_reduction * fossil_share_factor * efficiency_factor
            elif col in {"EF Cropland", "EF Grazing Land"}:
                modifier *= food_reduction
            ef_parts[col] = value * modifier

        ef_total = sum(ef_parts.values())

        ppm_for_damage = CARBON_PARAMS["baseline_ppm"] + (
            cumulative_atmospheric_gtco2 / CARBON_PARAMS["ppm_per_gtco2"]
        )
        temperature_for_damage = temp_from_ppm(ppm_for_damage)

        climate_damage = max(0.0, temperature_for_damage - 1.5) * (params["danno_clima"] / 100)
        biocap_total = base_biocap_total * biocap_growth * ((1 - climate_damage) ** step)
        biocap_total = max(biocap_total, 0.0)

        carbon_ratio = ef_parts.get("EF Carbon", 0.0) / base_carbon_ef
        area_emissions_factor = base_carbon_ef / world_reference_carbon_ef_gha if world_reference_carbon_ef_gha > 0 else 1.0
        gross_emissions = CARBON_PARAMS["baseline_emissions_gtco2"] * area_emissions_factor * carbon_ratio
        removed = params["rimozione_co2"] * progress
        net_emissions = max(0.0, gross_emissions - removed)

        cumulative_gross_emissions += gross_emissions
        cumulative_removed += removed
        cumulative_atmospheric_gtco2 += net_emissions * CARBON_PARAMS["airborne_fraction"]
        cumulative_atmospheric_without_removal_gtco2 += gross_emissions * CARBON_PARAMS["airborne_fraction"]

        ppm = CARBON_PARAMS["baseline_ppm"] + (
            cumulative_atmospheric_gtco2 / CARBON_PARAMS["ppm_per_gtco2"]
        )
        ppm_without_removal = CARBON_PARAMS["baseline_ppm"] + (
            cumulative_atmospheric_without_removal_gtco2 / CARBON_PARAMS["ppm_per_gtco2"]
        )
        temperature = temp_from_ppm(ppm)

        ratio_earths = ef_total / biocap_total if biocap_total > 0 else float("inf")
        saldo = biocap_total - ef_total

        rows.append(
            {
                "Anno": year,
                "Popolazione (Mld)": population / 1e9,
                "Biocapacita (Gha)": biocap_total,
                "Impronta Totale (Gha)": ef_total,
                "EF Carbon (Gha)": ef_parts.get("EF Carbon", 0.0),
                "EF Cropland (Gha)": ef_parts.get("EF Cropland", 0.0),
                "EF Grazing Land (Gha)": ef_parts.get("EF Grazing Land", 0.0),
                "EF Forest Products (Gha)": ef_parts.get("EF Forest Products", 0.0),
                "EF Fishing Grounds (Gha)": ef_parts.get("EF Fishing Grounds", 0.0),
                "EF Built-up Land (Gha)": ef_parts.get("EF Built-up Land", 0.0),
                "Saldo (Gha)": saldo,
                "Rapporto (Terre)": ratio_earths,
                "Emissioni Lorde (GtCO2/anno)": gross_emissions,
                "Rimozione CO2 (GtCO2/anno)": removed,
                "Emissioni Nette (GtCO2/anno)": net_emissions,
                "Emissioni Lorde Cumulative (GtCO2)": cumulative_gross_emissions,
                "CO2 Rimossa Cumulativa (GtCO2)": cumulative_removed,
                "CO2 Atmosferica Aggiuntiva (GtCO2)": cumulative_atmospheric_gtco2,
                "CO2 (ppm)": ppm,
                "CO2 senza rimozione (ppm)": ppm_without_removal,
                "Temperatura (C)": temperature,
                "Overshoot Day": overshoot_day(year, ratio_earths),
                "is_overshoot": ratio_earths > 1,
                "Energia fossile (%)": energy_fossil_share(energy_mix),
                "Petrolio (%)": energy_mix["Petrolio"],
                "Carbone (%)": energy_mix["Carbone"],
                "Gas naturale (%)": energy_mix["Gas naturale"],
                "Nucleare (%)": energy_mix["Nucleare"],
                "Idroelettrico (%)": energy_mix["Idroelettrico"],
                "Rinnovabili moderne (%)": energy_mix["Rinnovabili moderne"],
                "Bioenergie e altro (%)": energy_mix["Bioenergie e altro"],
                "Efficienza energia (%)": params["efficienza_energia"] * progress,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("Controlli scenario")

    if data_error:
        st.error(f"Dati ufficiali Footprint non caricati: {data_error}")

    st.selectbox(
        "Area dati Footprint",
        available_areas,
        index=available_areas.index(selected_area) if selected_area in available_areas else 0,
        key="area_dati",
        help="Default: Mondo. Seleziona Europa per immaginare un pianeta limitato alla biocapacita europea e al consumo degli europei.",
    )
    st.caption(
        "L'area selezionata cambia baseline, trend storici recenti, componenti dell'impronta e biocapacita."
    )

    if selected_area.lower() == "europa":
        st.selectbox(
            "Tipo scenario Europa",
            ["Trend recente", "Consumo costante"],
            key="tipo_scenario_europa",
            help="Trend recente estende i trend europei dei dati ufficiali Footprint. Consumo costante mantiene l'impronta europea iniziale costante prima delle leve.",
        )

    if effective_scenario_type == "Trend recente":
        st.caption(
            f"Trend recente {selected_area}: impronta {recent_trends['ef'] * 100:+.2f}%/anno, biocapacita {recent_trends['biocap'] * 100:+.2f}%/anno."
        )
        if selected_area.lower() == "europa" and recent_trends["ef"] < 0:
            st.caption(
                "Per Europa il trend recente e una decrescita dell'impronta: lo scenario puo apparire migliorativo pur partendo da un overshoot molto alto."
            )
    else:
        st.caption(
            "Consumo costante Europa: l'impronta iniziale resta stabile nel tempo; le variazioni arrivano solo da leve, popolazione e mix energia."
        )

    quality_col = get_quality_column(world_data) if not world_data.empty else None
    quality = baseline.get(quality_col, "n/d") if quality_col else "n/d"

    st.subheader(f"Baseline Footprint - {selected_area}")
    st.metric("Anno baseline", f"{int(baseline['Anno'])}")
    st.metric("Impronta", f"{float(baseline['EFConsTotGHA']) / 1e9:.2f} Gha")
    st.metric("Biocapacita", f"{float(baseline['BiocapTotGHA']) / 1e9:.2f} Gha")
    st.metric("Rapporto", f"{float(baseline['EFConsTotGHA']) / float(baseline['BiocapTotGHA']):.2f} Terre")
    st.caption(f"Qualita dato: {quality}")
    st.caption("EF = Ecological Footprint, cioe impronta ecologica.")
    st.caption("1 Gha = 1 miliardo di ettari = 10 milioni di km2.")

    st.divider()
    st.subheader("Scenari rapidi")
    scen_col1, scen_col2 = st.columns(2)
    scen_col1.button("BAU", on_click=apply_bau, width="stretch")
    scen_col2.button("Stress", on_click=apply_stress_ecologico, width="stretch")
    scen_col1.button("Transizione moderata", on_click=apply_transizione_moderata, width="stretch")
    scen_col2.button("Transizione forte", on_click=apply_transizione_forte, width="stretch")

    st.divider()
    st.subheader("BAU data-informed")
    st.caption(
        "Default calcolati dai dati ufficiali Footprint Mondo 2018-2026. Le leve modificano questa traiettoria."
    )

    st.session_state.trend_ef = st.slider(
        "Trend impronta totale (% annuo)",
        -2.0,
        3.0,
        value=float(st.session_state.trend_ef),
        step=0.05,
        format="%.2f%%",
    )
    st.session_state.trend_biocap = st.slider(
        "Trend biocapacita (% annuo)",
        -2.0,
        3.0,
        value=float(st.session_state.trend_biocap),
        step=0.05,
        format="%.2f%%",
    )
    st.session_state.deviazione_pop = st.slider(
        "Deviazione popolazione da ONU (% annuo)",
        -1.0,
        1.0,
        value=float(st.session_state.deviazione_pop),
        step=0.05,
        format="%.2f%%",
    )

    st.divider()
    st.subheader("Leve settoriali")

    st.session_state.leva_carbonio = st.slider(
        "Riduzione impronta carbonio entro il 2075 (%)",
        0.0,
        100.0,
        value=float(st.session_state.leva_carbonio),
        step=1.0,
        help="Esempio: 40% indica una forte riduzione della componente carbonio tramite rinnovabili, efficienza, elettrificazione e minori combustibili fossili.",
    )
    st.session_state.leva_dieta = st.slider(
        "Riduzione impronta agro-alimentare entro il 2075 (%)",
        0.0,
        80.0,
        value=float(st.session_state.leva_dieta),
        step=1.0,
        help="Agisce su impronta coltivazioni e pascolo. Esempio didattico: 30% puo essere letto come carne media da 43 a circa 30 kg/persona/anno.",
    )
    st.session_state.leva_forestazione = st.slider(
        "Extra crescita biocapacita da ripristino (% annuo)",
        0.0,
        1.5,
        value=float(st.session_state.leva_forestazione),
        step=0.05,
        format="%.2f%%",
        help="Esempio: +0.40% annuo sembra piccolo, ma accumulato fino al 2075 aumenta sensibilmente la biocapacita.",
    )
    st.session_state.rimozione_co2 = st.slider(
        "Rimozione CO2 al 2075 (GtCO2/anno)",
        0.0,
        20.0,
        value=float(st.session_state.rimozione_co2),
        step=0.5,
        help="La rimozione cresce linearmente da 0 nel 2026 al valore scelto nel 2075.",
    )

    st.divider()
    st.subheader("Energia")
    st.caption("Modulo semplificato ispirato a simulatori come EN-ROADS.")

    st.session_state.spinta_rinnovabili = st.slider(
        "Aumento rinnovabili moderne al 2075 (punti %)",
        0.0,
        70.0,
        value=float(st.session_state.spinta_rinnovabili),
        step=1.0,
        help="Aumenta eolico, solare e altre rinnovabili moderne, sottraendo quota a petrolio, carbone e gas.",
    )
    st.session_state.spinta_nucleare = st.slider(
        "Aumento nucleare al 2075 (punti %)",
        0.0,
        30.0,
        value=float(st.session_state.spinta_nucleare),
        step=1.0,
        help="Aumenta la quota nucleare, sottraendo quota a petrolio, carbone e gas.",
    )
    st.session_state.efficienza_energia = st.slider(
        "Efficienza / minore domanda energia al 2075 (%)",
        0.0,
        50.0,
        value=float(st.session_state.efficienza_energia),
        step=1.0,
        help="Riduce progressivamente la domanda energetica utile rispetto alla traiettoria BAU.",
    )
    energy_baseline_sidebar = get_energy_baseline(selected_area)
    fossil_baseline_sidebar = energy_fossil_share(energy_baseline_sidebar)
    low_carbon_baseline_sidebar = 100 - fossil_baseline_sidebar
    st.caption(
        f"Mix energia iniziale {selected_area}: fossili {fossil_baseline_sidebar:.1f}%, basse emissioni {low_carbon_baseline_sidebar:.1f}%."
    )
    if selected_area.lower() == "europa":
        st.caption(
            "Nota: l'Europa importa molta energia fossile. Qui il mix rappresenta il consumo energetico attribuito all'area, non l'autosufficienza produttiva."
        )

    st.divider()
    st.subheader("Feedback climatico")
    st.session_state.danno_clima = st.slider(
        "Penalita biocapacita per C sopra +1.5 (% annuo)",
        0.0,
        1.0,
        value=float(st.session_state.danno_clima),
        step=0.05,
        format="%.2f%%",
        help="Default 0: il clima non distrugge automaticamente la biocapacita. Aumentare solo per test di sensibilita.",
    )


params = {
    "trend_ef": st.session_state.trend_ef,
    "trend_biocap": st.session_state.trend_biocap,
    "leva_carbonio": st.session_state.leva_carbonio,
    "leva_dieta": st.session_state.leva_dieta,
    "leva_forestazione": st.session_state.leva_forestazione,
    "rimozione_co2": st.session_state.rimozione_co2,
    "spinta_rinnovabili": st.session_state.spinta_rinnovabili,
    "spinta_nucleare": st.session_state.spinta_nucleare,
    "efficienza_energia": st.session_state.efficienza_energia,
    "deviazione_pop": st.session_state.deviazione_pop,
    "danno_clima": st.session_state.danno_clima,
}

df = run_model(params)
df_display = df.rename(
    columns={
        "EF Carbon (Gha)": "Impronta carbonio (Gha)",
        "EF Cropland (Gha)": "Impronta coltivazioni (Gha)",
        "EF Grazing Land (Gha)": "Impronta pascolo (Gha)",
        "EF Forest Products (Gha)": "Impronta prodotti forestali (Gha)",
        "EF Fishing Grounds (Gha)": "Impronta pesca (Gha)",
        "EF Built-up Land (Gha)": "Impronta suolo costruito (Gha)",
    }
)
last = df.iloc[-1]
practical = practical_interpretation(params, baseline)
ratio_unit_label = "Terre" if selected_area.lower() == "mondo" else "Europe"
ratio_unit_singular = "Terra" if selected_area.lower() == "mondo" else "Europa"
ratio_display_col = f"Rapporto ({ratio_unit_label})"
df_display = df_display.rename(columns={"Rapporto (Terre)": ratio_display_col})


# ============================================================================
# DASHBOARD
# ============================================================================

st.subheader("Indicatori chiave 2075")
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.metric("Rapporto", f"{last['Rapporto (Terre)']:.2f} {ratio_unit_label}")
with kpi2:
    st.metric(
        "Impronta",
        f"{last['Impronta Totale (Gha)']:.2f} Gha",
        help="1 Gha = 1 miliardo di ettari = 10 milioni di km2.",
    )
with kpi3:
    st.metric(
        "Biocapacita",
        f"{last['Biocapacita (Gha)']:.2f} Gha",
        help="1 Gha = 1 miliardo di ettari = 10 milioni di km2.",
    )
with kpi4:
    st.metric("Emissioni nette", f"{last['Emissioni Nette (GtCO2/anno)']:.1f} GtCO2/anno")
with kpi5:
    st.metric("CO2 atmosferica", f"{last['CO2 (ppm)']:.0f} ppm")
with kpi6:
    st.metric("Temperatura", f"+{last['Temperatura (C)']:.2f} C")

st.info(f"Earth Overshoot Day 2075: {last['Overshoot Day']}")
if selected_area.lower() == "europa":
    st.warning(
        "Scenario Europa: il modello interpreta l'Europa come sistema chiuso, con biocapacita europea e consumi europei. "
        "E un esperimento didattico, non una previsione geopolitica."
    )
    if effective_scenario_type == "Trend recente" and recent_trends["ef"] < 0:
        st.info(
            f"Premessa importante: nei dati ufficiali Footprint recenti l'impronta europea e in decrescita ({recent_trends['ef'] * 100:+.2f}%/anno). "
            "Per vedere lo stress strutturale dei consumi europei, prova 'Consumo costante'."
        )
else:
    st.caption("Scenario Mondo: baseline e trend sono calcolati sui dati ufficiali Footprint globali.")

if selected_area.lower() == "mondo":
    st.subheader("Traduzione pratica delle leve")
    pr_col1, pr_col2, pr_col3, pr_col4 = st.columns(4)
    with pr_col1:
        st.metric(
            "Carne pro capite",
            f"{practical['meat_target']:.1f} kg/anno",
            f"-{practical['meat_reduction']:.1f} kg",
            help="Stima semplificata: interpreta la riduzione agro-alimentare come minore consumo medio di carne rispetto a 43 kg/persona/anno.",
        )
    with pr_col2:
        st.metric(
            "Decarb. effettiva stimata",
            f"{practical['effective_carbon_reduction']:.0f}%",
            f"energia {practical['energy_only_reduction']:.0f}%",
            help=(
                "Stima semplificata dell'effetto combinato su carbonio: riduzione impronta carbonio, "
                "sostituzione dei fossili con rinnovabili/nucleare ed efficienza energetica."
            ),
        )
    with pr_col3:
        st.metric(
            "Ripristino biocapacita",
            f"+{practical['extra_biocap_gha']:.2f} Gha",
            f"{practical['extra_biocap_km2_million']:.1f} mln km2",
            help="Effetto cumulato indicativo dell'extra crescita di biocapacita al 2075. 1 Gha = 10 milioni di km2.",
        )
    with pr_col4:
        st.metric(
            "CO2 rimossa",
            f"{practical['co2_removal_2075']:.1f} Gt/anno",
            f"{practical['co2_removal_cumulative']:.0f} Gt cum.",
            help="La rimozione cresce linearmente da 0 nel 2026 al valore scelto nel 2075.",
        )
    st.caption(
        "Queste equivalenze sono esempi didattici, non conversioni ufficiali. La decarbonizzazione stimata include anche rinnovabili, nucleare ed efficienza energetica."
    )
else:
    st.caption(
        "Le equivalenze pratiche delle leve sono nascoste nello scenario Europa per evitare interpretazioni non calibrate sui consumi medi europei."
    )

st.divider()

st.subheader("Biocapacita vs impronta")
fig_main = go.Figure()

if not world_data.empty:
    hist = world_data[world_data["Anno"].le(START_YEAR)].copy()
    fig_main.add_trace(
        go.Scatter(
            x=hist["Anno"],
            y=hist["BiocapTotGHA"] / 1e9,
            name="Biocapacita storica",
            line=dict(color="#059669", width=2),
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=hist["Anno"],
            y=hist["EFConsTotGHA"] / 1e9,
            name="Impronta storica",
            line=dict(color="#dc2626", width=2),
        )
    )

fig_main.add_trace(
    go.Scatter(
        x=df["Anno"],
        y=df["Biocapacita (Gha)"],
        name="Biocapacita proiettata",
        line=dict(color="#10b981", width=3, dash="dash"),
    )
)
fig_main.add_trace(
    go.Scatter(
        x=df["Anno"],
        y=df["Impronta Totale (Gha)"],
        name="Impronta proiettata",
        line=dict(color="#ef4444", width=3, dash="dash"),
    )
)
fig_main.update_layout(
    xaxis_title="Anno",
    yaxis_title="Gha",
    hovermode="x unified",
    height=440,
    template="plotly_white",
)
st.plotly_chart(fig_main, width="stretch")

st.caption(
    "L'equilibrio non e una soglia fissa a 1 Gha: avviene quando la linea dell'impronta incontra la linea della biocapacita."
)


st.subheader("Consumo del pianeta")
fig_ratio = go.Figure()

if not world_data.empty:
    fig_ratio.add_trace(
        go.Scatter(
            x=world_data["Anno"],
            y=world_data["Rapporto Area"],
            name=f"{ratio_unit_label} storiche",
            line=dict(color="#7c3aed", width=2),
        )
    )

fig_ratio.add_trace(
    go.Scatter(
        x=df["Anno"],
        y=df["Rapporto (Terre)"],
        name=f"{ratio_unit_label} proiettate",
        line=dict(color="#f59e0b", width=3, dash="dash"),
    )
)
fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text=f"1 {ratio_unit_singular}")
fig_ratio.add_vline(
    x=START_YEAR,
    line_dash="dot",
    line_color="rgba(100, 116, 139, 0.7)",
    annotation_text="inizio proiezione",
)
fig_ratio.update_layout(
    xaxis_title="Anno",
    yaxis_title=f"Numero di {ratio_unit_label}",
    hovermode="x unified",
    height=360,
    template="plotly_white",
)
st.plotly_chart(fig_ratio, width="stretch")
st.caption(
    f"Linea viola: dati ufficiali Footprint storici. Linea arancione tratteggiata: proiezione del modello dal {START_YEAR} al {END_YEAR}."
)
if selected_area.lower() == "europa":
    st.caption(
        "Nota Europa: il grafico mostra quante Europe servirebbero per sostenere i consumi europei "
        "(impronta totale europea / biocapacita europea). Questo e diverso da 'Numero di Terre consumate', "
        "che indica quante Terre servirebbero se tutti vivessero con il consumo medio europeo."
    )


st.subheader("Emissioni CO2 e concentrazione atmosferica")
fig_carbon = make_subplots(specs=[[{"secondary_y": True}]])
fig_carbon.add_trace(
    go.Bar(
        x=df["Anno"],
        y=df["Emissioni Lorde (GtCO2/anno)"],
        name="Emissioni lorde",
        marker_color="rgba(239, 68, 68, 0.35)",
        opacity=0.75,
    ),
    secondary_y=False,
)
fig_carbon.add_trace(
    go.Scatter(
        x=df["Anno"],
        y=df["Emissioni Nette (GtCO2/anno)"],
        name="Emissioni nette",
        mode="lines",
        line=dict(color="#0ea5e9", width=3),
    ),
    secondary_y=False,
)
if float(df["Rimozione CO2 (GtCO2/anno)"].max()) > 0:
    fig_carbon.add_trace(
        go.Bar(
            x=df["Anno"],
            y=-df["Rimozione CO2 (GtCO2/anno)"],
            name="CO2 rimossa",
            marker_color="rgba(16, 185, 129, 0.45)",
            opacity=0.8,
        ),
        secondary_y=False,
    )
fig_carbon.add_trace(
    go.Scatter(
        x=df["Anno"],
        y=df["CO2 (ppm)"],
        name="CO2 ppm",
        line=dict(color="#111827", width=5),
        mode="lines+markers",
        marker=dict(size=4),
    ),
    secondary_y=True,
)
if float(df["Rimozione CO2 (GtCO2/anno)"].max()) > 0:
    fig_carbon.add_trace(
        go.Scatter(
            x=df["Anno"],
            y=df["CO2 senza rimozione (ppm)"],
            name="CO2 ppm senza rimozione",
            line=dict(color="rgba(17, 24, 39, 0.45)", width=3, dash="dash"),
            mode="lines",
        ),
        secondary_y=True,
    )
ppm_min = float(min(df["CO2 (ppm)"].min(), df["CO2 senza rimozione (ppm)"].min()))
ppm_max = float(max(df["CO2 (ppm)"].max(), df["CO2 senza rimozione (ppm)"].max()))
ppm_padding = max(5.0, (ppm_max - ppm_min) * 0.12)
fig_carbon.update_yaxes(title_text="GtCO2/anno", secondary_y=False)
fig_carbon.update_yaxes(
    title_text="ppm CO2",
    secondary_y=True,
    range=[ppm_min - ppm_padding, ppm_max + ppm_padding],
    showgrid=False,
)
fig_carbon.update_layout(
    xaxis_title="Anno",
    hovermode="x unified",
    height=420,
    template="plotly_white",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_carbon, width="stretch")

st.caption(
    "Le emissioni sono flussi annuali; i ppm rappresentano lo stock atmosferico cumulato. Il modello usa CO2, non CO2e."
)
if float(df["Rimozione CO2 (GtCO2/anno)"].max()) > 0:
    ppm_avoided = last["CO2 senza rimozione (ppm)"] - last["CO2 (ppm)"]
    st.caption(
        f"Effetto rimozione CO2 nello scenario: circa {ppm_avoided:.1f} ppm evitati al 2075 rispetto allo stesso scenario senza rimozione."
    )


st.subheader("Mix energetico globale semplificato")
fig_energy = go.Figure()
energy_cols = {
    "Petrolio (%)": "#92400e",
    "Carbone (%)": "#1f2937",
    "Gas naturale (%)": "#64748b",
    "Nucleare (%)": "#a855f7",
    "Idroelettrico (%)": "#0ea5e9",
    "Rinnovabili moderne (%)": "#22c55e",
    "Bioenergie e altro (%)": "#84cc16",
}
for col, color in energy_cols.items():
    fig_energy.add_trace(
        go.Scatter(
            x=df["Anno"],
            y=df[col],
            name=col.replace(" (%)", ""),
            stackgroup="energy",
            line=dict(color=color),
        )
    )
fig_energy.update_layout(
    xaxis_title="Anno",
    yaxis_title="Quota energia primaria (%)",
    hovermode="x unified",
    height=420,
    template="plotly_white",
)
st.plotly_chart(fig_energy, width="stretch")
st.caption(
    f"Modulo semplificato basato sul mix iniziale {selected_area}: rinnovabili e nucleare sostituiscono progressivamente petrolio, carbone e gas; l'efficienza riduce la domanda energetica."
)

energy_kpi1, energy_kpi2, energy_kpi3, energy_kpi4 = st.columns(4)
with energy_kpi1:
    st.metric("Fossili 2075", f"{last['Energia fossile (%)']:.1f}%")
with energy_kpi2:
    low_carbon_2075 = sum(last[f"{source} (%)"] for source in LOW_CARBON_ENERGY_SOURCES)
    st.metric("Basse emissioni 2075", f"{low_carbon_2075:.1f}%")
with energy_kpi3:
    st.metric("Rinnovabili moderne 2075", f"{last['Rinnovabili moderne (%)']:.1f}%")
with energy_kpi4:
    st.metric("Nucleare 2075", f"{last['Nucleare (%)']:.1f}%")


st.subheader("Composizione dell'impronta ecologica")
fig_parts = go.Figure()
part_map = {
    "EF Carbon (Gha)": ("Carbonio", "#6366f1"),
    "EF Cropland (Gha)": ("Coltivazioni", "#84cc16"),
    "EF Grazing Land (Gha)": ("Pascolo", "#f59e0b"),
    "EF Forest Products (Gha)": ("Prodotti forestali", "#10b981"),
    "EF Fishing Grounds (Gha)": ("Pesca", "#0ea5e9"),
    "EF Built-up Land (Gha)": ("Suolo costruito", "#64748b"),
}
for col, (label, color) in part_map.items():
    fig_parts.add_trace(
        go.Scatter(
            x=df["Anno"],
            y=df[col],
            name=label,
            stackgroup="one",
            line=dict(color=color),
        )
    )
fig_parts.update_layout(
    xaxis_title="Anno",
    yaxis_title="Gha",
    hovermode="x unified",
    height=420,
    template="plotly_white",
)
st.plotly_chart(fig_parts, width="stretch")


st.divider()
st.subheader("Dati sintetici")
display_cols = [
    "Anno",
    "Popolazione (Mld)",
    "Impronta Totale (Gha)",
    "Biocapacita (Gha)",
    "Impronta carbonio (Gha)",
    "Impronta coltivazioni (Gha)",
    "Impronta pascolo (Gha)",
    ratio_display_col,
    "Rimozione CO2 (GtCO2/anno)",
    "Emissioni Nette (GtCO2/anno)",
    "CO2 (ppm)",
    "Temperatura (C)",
    "Energia fossile (%)",
    "Rinnovabili moderne (%)",
    "Nucleare (%)",
    "Overshoot Day",
]
st.dataframe(
    df_display[display_cols].round(3),
    width="stretch",
    hide_index=True,
)


st.divider()
st.markdown(
    """
    <div style="border-left: 5px solid #f59e0b; padding: 0.35rem 0 0.35rem 0.8rem; margin-bottom: 0.35rem;">
        <h3 style="color: #f59e0b; margin: 0;">Approfondisci con IA</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Modulo opzionale: usa Gemini per domande generali o per spiegare aspetti dello scenario. L'IA non modifica i calcoli del simulatore."
)

if "ai_question" not in st.session_state:
    st.session_state.ai_question = (
        "Il consumo medio di carne indicato nel modello come puo essere suddiviso tra bovino, suino, pollame e altri tipi?"
    )
if "ai_last_answer" not in st.session_state:
    st.session_state.ai_last_answer = ""
if "ai_last_question" not in st.session_state:
    st.session_state.ai_last_question = ""
if "ai_history" not in st.session_state:
    st.session_state.ai_history = []
if st.session_state.get("ai_clear_question"):
    st.session_state.ai_question = ""
    st.session_state.ai_clear_question = False

with st.expander("Chiedi un approfondimento"):
    examples = st.columns(3)
    if examples[0].button("Carne e dieta", width="stretch"):
        st.session_state.ai_question = "Spiega in modo semplice come interpretare la leva agro-alimentare e quali dati servirebbero per distinguere bovino, suino, pollame e altri tipi di carne."
    if examples[1].button("Energia", width="stretch"):
        st.session_state.ai_question = "Spiega il ruolo di petrolio, gas, carbone, rinnovabili e nucleare nello scenario corrente."
    if examples[2].button("Europa", width="stretch"):
        st.session_state.ai_question = "Spiega perche lo scenario Europa puo risultare piu critico o piu migliorativo a seconda del tipo scenario scelto."

    ai_col1, ai_col2 = st.columns([2, 1])
    with ai_col1:
        ai_question = st.text_area(
            "Domanda",
            key="ai_question",
            height=110,
            help="Esempio: chiedi chiarimenti su alimentazione, energia, CO2, Europa/Mondo o risultati dello scenario.",
        )
    with ai_col2:
        ai_model_choice = st.selectbox(
            "Modello Gemini",
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "Personalizzato",
            ],
            help="Flash e piu rapido/economico; Pro e piu adatto a ragionamenti lunghi.",
        )
        custom_ai_model = ""
        if ai_model_choice == "Personalizzato":
            custom_ai_model = st.text_input("Nome modello personalizzato", value="")
        manual_api_key = st.text_input(
            "Google/Gemini API key",
            type="password",
            help=(
                "Puoi creare una API key gratuita in Google AI Studio: vai su "
                "https://aistudio.google.com/app/apikey, accedi con Google, scegli Create API key, "
                "copia la chiave e incollala qui. In alternativa configura GEMINI_API_KEY o GOOGLE_API_KEY."
            ),
        )
        st.caption("La chiave inserita qui non viene salvata nel codice.")
        with st.expander("Come creare una API key gratuita"):
            st.markdown(
                """
                1. Vai su [Google AI Studio](https://aistudio.google.com/app/apikey).
                2. Accedi con il tuo account Google.
                3. Clicca **Create API key**.
                4. Copia la chiave generata.
                5. Incollala nel campo password qui sopra.

                Nota: le quote gratuite e le condizioni possono cambiare nel tempo. Controlla sempre i limiti indicati da Google.
                """
            )

    if st.button("Genera approfondimento IA", width="stretch"):
        api_key = get_configured_gemini_key(manual_api_key)
        if not api_key:
            st.warning(
                "Configura una chiave Gemini: inseriscila nel campo password oppure imposta GEMINI_API_KEY/GOOGLE_API_KEY in Streamlit secrets o nelle variabili ambiente."
            )
        elif not ai_question.strip():
            st.warning("Inserisci una domanda.")
        else:
            try:
                with st.spinner("Sto generando l'approfondimento..."):
                    selected_ai_model = (
                        custom_ai_model.strip()
                        if ai_model_choice == "Personalizzato" and custom_ai_model.strip()
                        else ai_model_choice
                    )
                    answer = ask_gemini(ai_question, api_key, selected_ai_model)
                st.session_state.ai_last_question = ai_question
                st.session_state.ai_last_answer = answer
                st.session_state.ai_history.insert(
                    0,
                    {
                        "question": ai_question,
                        "answer": answer,
                        "model": selected_ai_model,
                        "area": selected_area,
                    },
                )
                st.session_state.ai_history = st.session_state.ai_history[:10]
                st.session_state.ai_clear_question = True
                st.rerun()
            except Exception as exc:
                st.error(f"Errore nella chiamata IA: {exc}")

    if st.session_state.ai_last_answer:
        st.markdown("#### Risposta IA")
        st.markdown(st.session_state.ai_last_answer)

        ai_export = (
            "# Approfondimento IA\n\n"
            f"## Domanda\n\n{st.session_state.ai_last_question}\n\n"
            f"## Risposta\n\n{st.session_state.ai_last_answer}\n"
        )
        copy_col, save_col = st.columns([1, 1])
        with copy_col:
            with st.expander("Copia risposta"):
                st.caption("Usa l'icona di copia del blocco qui sotto.")
                st.code(st.session_state.ai_last_answer, language="markdown")
        with save_col:
            st.download_button(
                "Scarica risposta IA (.md)",
                data=ai_export.encode("utf-8"),
                file_name="approfondimento_ia.md",
                mime="text/markdown",
                width="stretch",
            )

    if st.session_state.ai_history:
        with st.expander("Storico domande IA"):
            history_export_parts = ["# Storico approfondimenti IA\n"]
            for idx, item in enumerate(st.session_state.ai_history, start=1):
                st.markdown(f"**{idx}. {item['question']}**")
                st.caption(f"Area: {item['area']} | Modello: {item['model']}")
                st.markdown(item["answer"])
                st.divider()
                history_export_parts.append(
                    f"## {idx}. Domanda\n\n{item['question']}\n\n"
                    f"Area: {item['area']}  \nModello: {item['model']}\n\n"
                    f"### Risposta\n\n{item['answer']}\n"
                )

            st.download_button(
                "Scarica storico IA (.md)",
                data="\n".join(history_export_parts).encode("utf-8"),
                file_name="storico_approfondimenti_ia.md",
                mime="text/markdown",
                width="stretch",
            )


st.divider()
st.subheader("Note metodologiche")
with st.expander("Che cosa fa il modello, in parole semplici", expanded=True):
    st.markdown(
        """
        Il modello prova a rispondere a una domanda:

        **se continuiamo con una certa traiettoria di consumo e rigenerazione, quante Terre servono ogni anno e che cosa succede a CO2 e temperatura?**

        Per farlo tiene separati tre livelli:

        - **Impronta ecologica**: quanta biocapacita consumiamo. Arriva dai dati ufficiali Footprint e viene misurata in Gha.
        - **Biocapacita**: quanta capacita rigenerativa offre il pianeta. Anche questa arriva dai dati ufficiali Footprint.
        - **Carbonio e clima**: le emissioni annue aumentano la CO2 atmosferica, la CO2 in ppm influenza la temperatura.

        Esempio: se nel 2026 il rapporto e circa **1,78 Terre**, significa che l'umanita consuma in un anno risorse che richiederebbero circa 1,78 pianeti per essere rigenerate nello stesso anno.

        Acronimi usati:

        - **EF** = *Ecological Footprint*, cioe impronta ecologica.
        - **Gha** = giga-ettari globali. **1 Gha = 1 miliardo di ettari = 10 milioni di km2**.
        - **BAU** = *Business As Usual*, scenario in cui si prosegue secondo la traiettoria corrente.
        - **ppm** = parti per milione, unita usata per la concentrazione di CO2 in atmosfera.
        - **GtCO2** = miliardi di tonnellate di CO2.
        """
    )

with st.expander("Mondo o Europa: come interpretare la scelta dell'area"):
    st.markdown(
        """
        Il selettore **Area dati Footprint** cambia il sistema di riferimento del modello.

        **Mondo** e il default:

        ```text
        biocapacita mondiale
        impronta ecologica mondiale
        consumo medio globale
        ```

        **Europa** risponde a una domanda diversa:

        ```text
        se il pianeta disponibile fosse l'Europa
        e il consumo fosse quello degli europei,
        quante "Europe" servirebbero?
        ```

        In questo caso il modello usa:

        - la biocapacita totale europea;
        - l'impronta totale europea;
        - le componenti europee dell'impronta;
        - i trend recenti europei nei dati ufficiali Footprint.
        - un mix energetico europeo semplificato, diverso dalla media mondiale.

        Attenzione: non e una previsione politica o economica sull'Europa reale. E un esperimento di scala: restringe il sistema alla biocapacita europea e misura se il consumo europeo e compatibile con quella capacita.

        C'e inoltre un limite importante: l'Europa importa una quota rilevante di combustibili fossili e materie prime energetiche. Quindi il modello non dice "l'Europa produce tutta questa energia dentro i suoi confini"; dice piuttosto: "questa e l'impronta attribuita al consumo europeo".

        Il selettore **Tipo scenario base** serve proprio a evitare interpretazioni ambigue:

        - **Trend recente**: estende il trend osservato nei dati ufficiali Footprint. Per l'Europa questo trend recente e una decrescita dell'impronta, quindi la curva puo migliorare.
        - **Consumo costante**: mantiene stabile l'impronta iniziale dell'area. Per l'Europa e utile per visualizzare quanto e pesante partire da circa 3,1 Europe anche senza peggiorare ulteriormente i consumi.
        """
    )

with st.expander("Come viene calcolata la traiettoria"):
    st.markdown(
        """
        Il modello parte dai dati ufficiali Footprint `Mondo` piu recenti e poi proietta ogni anno fino al 2075.

        La formula centrale e:

        ```text
        Rapporto Terre = Impronta totale / Biocapacita
        ```

        Se il rapporto e:

        - **uguale a 1**: il pianeta rigenera quanto consumiamo;
        - **maggiore di 1**: siamo in overshoot;
        - **minore di 1**: siamo sotto la capacita rigenerativa annuale.

        Il BAU iniziale e ricavato dai dati ufficiali Footprint recenti:

        ```text
        Impronta:    crescita annua osservata recente
        Biocapacita: crescita annua osservata recente
        ```

        Le leve utente non sostituiscono i dati storici: modificano la traiettoria futura.
        """
    )

with st.expander("Regole matematiche del modello"):
    st.markdown(
        """
        Questa sezione descrive il nucleo tecnico del modello. Le formule sono volutamente semplici: l'obiettivo non e riprodurre un modello climatico completo, ma mantenere una catena causale coerente e controllabile.

        **1. Proiezione dell'impronta ecologica totale**

        Il modello parte dall'impronta ufficiale Footprint nel 2026 e la fa evolvere con crescita composta:

        ```text
        Impronta_t = Impronta_2026 * (1 + trend_impronta)^anni
        ```

        Poi la scompone nelle categorie ufficiali Footprint:

        ```text
        carbonio, coltivazioni, pascolo, prodotti forestali, pesca, suolo costruito
        ```

        Le leve non agiscono tutte sull'impronta totale indistintamente:

        - la leva carbonio modifica la componente carbonio;
        - la leva agro-alimentare modifica coltivazioni e pascolo;
        - il modulo energia modifica soprattutto la componente carbonio.

        **2. Le leve crescono gradualmente**

        Le leve sono applicate in modo progressivo dal 2026 al 2075:

        ```text
        progresso = (anno - 2026) / (2075 - 2026)
        effetto_anno = effetto_finale * progresso
        ```

        Questo evita salti irreali: una transizione del 50% al 2075 non avviene tutta nel 2026, ma cresce anno dopo anno.

        **3. Biocapacita**

        Anche la biocapacita usa crescita composta:

        ```text
        Biocap_t = Biocap_2026 * (1 + trend_biocap + ripristino)^anni
        ```

        Il parametro di ripristino rappresenta interventi aggiuntivi: riforestazione, recupero suoli, protezione ecosistemi, gestione piu sostenibile di pesca e foreste.

        **4. Rapporto Terre e Overshoot**

        La relazione centrale resta:

        ```text
        Rapporto Terre = Impronta_t / Biocap_t
        ```

        L'Overshoot Day viene stimato come:

        ```text
        giorni disponibili = 365 / Rapporto Terre
        ```

        Se il rapporto e 2, la biocapacita annuale viene consumata circa a meta anno.

        **5. Modulo energia**

        Il mix energetico parte da una baseline semplificata coerente con l'area selezionata:

        ```text
        petrolio + carbone + gas = quota fossile
        nucleare + idroelettrico + rinnovabili + bioenergie = quota a basse emissioni
        ```

        Quando aumentano rinnovabili o nucleare, il modello sottrae quella quota in modo proporzionale ai combustibili fossili:

        ```text
        nuova_quota_fossile = quota_fossile_base - sostituzione
        ```

        La componente carbonio dell'impronta viene poi corretta con:

        ```text
        fattore_fossile = quota_fossile_attuale / quota_fossile_base
        fattore_efficienza = 1 - efficienza_energia
        impronta_carbonio_t *= fattore_fossile * fattore_efficienza
        ```

        Motivazione: se il sistema energetico usa meno fossili e consuma meno energia utile, produce meno pressione carbonica.

        **6. Emissioni CO2, ppm e temperatura**

        Le emissioni lorde sono stimate dalla componente carbonio:

        ```text
        emissioni_lorde_t = emissioni_base * (impronta_carbonio_t / impronta_carbonio_2026)
        ```

        La rimozione CO2 e sottratta per ottenere emissioni nette:

        ```text
        emissioni_nette_t = max(0, emissioni_lorde_t - rimozione_t)
        ```

        Solo una parte delle emissioni nette resta in atmosfera. Il modello usa una frazione atmosferica media:

        ```text
        CO2_atmosferica_aggiuntiva += emissioni_nette_t * airborne_fraction
        ppm_t = ppm_2026 + CO2_atmosferica_aggiuntiva / 7.82
        ```

        dove `7.82 GtCO2` corrispondono circa a `1 ppm`.

        La temperatura non cresce linearmente con i ppm: usa una risposta logaritmica semplificata:

        ```text
        T_t = T_2026 + ECS * log2(ppm_t / ppm_2026)
        ```

        dove `ECS` e la sensibilita climatica al raddoppio della CO2. Questa scelta e piu coerente della vecchia formula lineare e impedisce che la CO2 venga trasformata direttamente in un collasso artificiale della biocapacita.
        """
    )

with st.expander("Significato dei parametri nella barra laterale"):
    st.markdown(
        """
        **Trend impronta totale (% annuo)**

        Indica quanto cresce o diminuisce ogni anno la domanda complessiva di risorse.

        Esempio: se lo imposti a `+0,40%`, l'impronta aumenta lentamente ogni anno. Se lo imposti a `-0,50%`, stai simulando una riduzione strutturale del consumo globale di risorse.

        **Trend biocapacita (% annuo)**

        Indica quanto cambia ogni anno la capacita rigenerativa del pianeta.

        Esempio: `+0,20%` significa lieve miglioramento o stabilita produttiva; `-0,40%` simula degrado progressivo di suoli, foreste, mari e capacita biologica.

        **Deviazione popolazione da ONU (% annuo)**

        Il modello usa una traiettoria ONU semplificata. Questo parametro permette di deviare da quella traiettoria.

        Esempio: `0%` segue la proiezione di base. `-0,20%` simula una popolazione futura un po' piu bassa del previsto. `+0,20%` simula una popolazione piu alta.

        Nota: il trend dell'impronta totale deriva gia dai dati Footprint complessivi. Per questo la popolazione non viene conteggiata due volte: modifica l'impronta solo quando imposti una deviazione diversa da zero.

        **Aumento rinnovabili moderne al 2075 (punti %)**

        Aumenta la quota di solare, eolico e altre rinnovabili moderne nel mix energetico globale. Nel modello questa quota viene sottratta proporzionalmente a petrolio, carbone e gas.

        Esempio: `+30 punti %` significa che entro il 2075 le rinnovabili moderne occupano 30 punti percentuali in piu del mix energetico rispetto alla baseline semplificata.

        **Aumento nucleare al 2075 (punti %)**

        Aumenta la quota nucleare nel mix energetico. Anche questa quota sostituisce progressivamente fonti fossili.

        Esempio: `+10 punti %` indica una forte espansione del nucleare globale entro il 2075.

        **Efficienza / minore domanda energia al 2075 (%)**

        Riduce la domanda energetica utile rispetto alla traiettoria BAU. Non cambia direttamente il mix, ma riduce il carico energetico che genera impronta carbonio.

        Esempio: `25%` significa che nel 2075 il sistema richiede il 25% di energia in meno rispetto allo scenario senza efficienza aggiuntiva.
        """
    )

with st.expander("Come usare le leve settoriali"):
    st.markdown(
        """
        **Riduzione impronta carbonio entro il 2075 (%)**

        Riduce progressivamente la componente carbonio dell'impronta ecologica e, nello stesso tempo, riduce le emissioni lorde di CO2.

        Esempio: `50%` significa che entro il 2075 la componente carbonio dell'impronta e dimezzata rispetto alla traiettoria BAU. Valori alti rappresentano transizione energetica, efficienza, elettrificazione e decarbonizzazione.

        **Riduzione impronta agro-alimentare entro il 2075 (%)**

        Agisce su coltivazioni e pascoli.

        Esempio: `30%` puo rappresentare meno spreco alimentare, dieta con meno carne, agricoltura piu efficiente e minore pressione sui pascoli.

        **Extra crescita biocapacita da ripristino (% annuo)**

        Aumenta la biocapacita futura rispetto al BAU.

        Esempio: `+0,30%` annuo rappresenta interventi costanti di ripristino: riforestazione, recupero suoli, protezione ecosistemi, pesca piu sostenibile.

        **Rimozione CO2 al 2075 (GtCO2/anno)**

        Simula una rimozione artificiale o aggiuntiva di CO2 che cresce gradualmente da 0 nel 2026 fino al valore scelto nel 2075.

        Esempio: `5 GtCO2/anno` significa che nel 2075 il sistema rimuove 5 miliardi di tonnellate di CO2 all'anno. Questa leva riduce le emissioni nette e rallenta la crescita dei ppm.
        """
    )

if selected_area.lower() == "mondo":
    with st.expander("Esempi pratici: come interpretare le leve"):
        st.markdown(
            f"""
            Questa sezione traduce le leve in esempi concreti. L'ispirazione e la logica dei simulatori come EN-ROADS: poche leve globali, risposta immediata, lettura pratica dello scenario.

            **1. Riduzione impronta agro-alimentare**

            Nel modello agisce su coltivazioni e pascoli. Una lettura semplice e:

            ```text
            0%   -> consumo medio carne circa 43 kg/persona/anno
            30%  -> circa 30 kg/persona/anno
            50%  -> circa 22 kg/persona/anno
            ```

            Nello scenario corrente equivale a circa **{practical['meat_target']:.1f} kg/persona/anno**, cioe **-{practical['meat_reduction']:.1f} kg** rispetto alla baseline didattica.

            **2. Riduzione impronta carbonio**

            Nel modello riduce la componente carbonio dell'impronta e le emissioni lorde. Non e una quota ufficiale di rinnovabili, ma si puo leggere cosi:

            ```text
            40% -> forte elettrificazione, efficienza e crescita rinnovabili
            70% -> trasformazione molto profonda del sistema energetico
            100% -> componente carbonio quasi azzerata nello scenario
            ```

            Nello scenario corrente la leva diretta di riduzione carbonio e **{practical['carbon_reduction']:.0f}%**, ma l'effetto stimato complessivo su carbonio, includendo mix energetico ed efficienza, e circa **{practical['effective_carbon_reduction']:.0f}%**.

            **3. Extra crescita biocapacita**

            Rappresenta ripristino di ecosistemi, suoli, foreste, zone di pesca e capacita biologica complessiva.

            Esempio: `+0,40%` annuo sembra poco, ma su quasi 50 anni si accumula. Nello scenario corrente vale circa **+{practical['extra_biocap_gha']:.2f} Gha** al 2075, cioe circa **{practical['extra_biocap_km2_million']:.1f} milioni di km2 equivalenti**.

            **4. Rimozione CO2**

            Indica quanta CO2 viene rimossa ogni anno nel 2075. Il modello la fa crescere gradualmente da zero.

            Esempio: `5 GtCO2/anno` al 2075 non significa 5 Gt ogni anno dal 2026: significa arrivare progressivamente a quel livello. Nello scenario corrente la rimozione cumulata indicativa e circa **{practical['co2_removal_cumulative']:.0f} GtCO2**.
            """
        )

with st.expander("CO2, ppm e temperatura: perche sono separati"):
    st.markdown(
        """
        Il modello distingue due cose diverse:

        - **Emissioni annue**, misurate in `GtCO2/anno`: sono il flusso prodotto ogni anno.
        - **CO2 atmosferica**, misurata in `ppm`: e lo stock accumulato in atmosfera.

        Esempio: anche se le emissioni annue smettono di crescere, la CO2 in atmosfera puo continuare ad aumentare se le emissioni nette restano positive.

        La temperatura viene stimata dalla CO2 atmosferica con una formula logaritmica semplificata. Non e un modello climatico completo, ma evita l'errore del vecchio modello, dove la CO2 non assorbita veniva trasformata direttamente in un'impronta enorme e causava risultati irreali.
        """
    )

with st.expander("Modulo energia: come funziona"):
    st.markdown(
        """
        La sezione energia e una versione molto semplificata dell'approccio usato da simulatori come EN-ROADS: l'utente modifica alcune grandi leve e osserva come cambiano emissioni, CO2 e temperatura.

        Il modello parte da un mix globale indicativo:

        ```text
        Petrolio, carbone e gas      -> fonti fossili
        Nucleare, idroelettrico,
        rinnovabili moderne e altro  -> fonti a basse emissioni
        ```

        Le leve energetiche fanno tre cose:

        - **Aumento rinnovabili moderne**: aumenta solare, eolico e rinnovabili moderne, riducendo petrolio, carbone e gas.
        - **Aumento nucleare**: aumenta la quota nucleare, riducendo petrolio, carbone e gas.
        - **Efficienza / minore domanda**: riduce il fabbisogno energetico rispetto alla traiettoria BAU.

        Esempio: se imposti `+30 punti %` di rinnovabili, il modello interpreta lo scenario come una forte sostituzione dei fossili con energia rinnovabile entro il 2075.

        Questo modulo non distingue ancora trasporti, industria, edifici ed elettricita. Serve a rendere leggibile il legame:

        ```text
        mix energetico -> impronta carbonio -> emissioni CO2 -> ppm -> temperatura
        ```
        """
    )

with st.expander("Esempi di scenari da provare"):
    st.markdown(
        """
        **Scenario BAU**

        Lascia i valori predefiniti. Serve come riferimento.

        **Scenario transizione moderata**

        ```text
        Riduzione impronta carbonio: 40%
        Riduzione impronta agro-alimentare: 20%
        Aumento rinnovabili moderne: +25 punti %
        Aumento nucleare: +5 punti %
        Efficienza energia: 15%
        Extra biocapacita: +0,20% annuo
        Rimozione CO2: 2 GtCO2/anno
        ```

        **Scenario transizione forte**

        ```text
        Riduzione impronta carbonio: 70%
        Riduzione impronta agro-alimentare: 35%
        Aumento rinnovabili moderne: +45 punti %
        Aumento nucleare: +10 punti %
        Efficienza energia: 25%
        Extra biocapacita: +0,40% annuo
        Rimozione CO2: 5 GtCO2/anno
        ```

        **Scenario stress ecologico**

        ```text
        Trend impronta: +0,80% annuo
        Trend biocapacita: -0,30% annuo
        Riduzione impronta carbonio: 0%
        Riduzione impronta agro-alimentare: 0%
        ```

        Confronta soprattutto tre risultati: `Rapporto`, `CO2 (ppm)` e `Temperatura (C)`.
        """
    )

with st.expander("Limiti attuali del modello"):
    st.markdown(
        """
        Questo modello e piu solido del precedente, ma resta una simulazione semplificata.

        - Le emissioni sono espresse come **CO2**, non come CO2e.
        - Il legame tra impronta carbonio e `GtCO2` e calibrato su un valore globale di riferimento, non su una serie storica completa.
        - La temperatura e una stima didattica, non una previsione climatica ufficiale.
        - Il danno climatico sulla biocapacita e disattivato di default per evitare collassi non calibrati.
        - Le leve rappresentano direzioni di scenario, non politiche fisiche dettagliate.

        Il prossimo miglioramento naturale sarebbe integrare una serie storica esterna di emissioni CO2 e ppm, per calibrare meglio il modulo carbonio.
        """
    )


st.divider()
csv_data = df.to_csv(index=False).encode("utf-8")
summary = f"""SCENARIO QUANTITATIVO 2075
===========================
Area dati: {selected_area}
Tipo scenario base: {effective_scenario_type}
Trend impronta: {params['trend_ef']:.2f}% annuo
Trend biocapacita: {params['trend_biocap']:.2f}% annuo
Riduzione impronta carbonio al 2075: {params['leva_carbonio']:.1f}%
Riduzione impronta agro-alimentare al 2075: {params['leva_dieta']:.1f}%
Extra crescita biocapacita: {params['leva_forestazione']:.2f}% annuo
Rimozione CO2 al 2075: {params['rimozione_co2']:.1f} GtCO2/anno
Aumento rinnovabili moderne al 2075: {params['spinta_rinnovabili']:.1f} punti %
Aumento nucleare al 2075: {params['spinta_nucleare']:.1f} punti %
Efficienza energia al 2075: {params['efficienza_energia']:.1f}%

Risultati 2075:
Impronta: {last['Impronta Totale (Gha)']:.2f} Gha
Biocapacita: {last['Biocapacita (Gha)']:.2f} Gha
Rapporto: {last['Rapporto (Terre)']:.2f} {ratio_unit_label}
Emissioni nette: {last['Emissioni Nette (GtCO2/anno)']:.1f} GtCO2/anno
Energia fossile: {last['Energia fossile (%)']:.1f}%
Rinnovabili moderne: {last['Rinnovabili moderne (%)']:.1f}%
Nucleare: {last['Nucleare (%)']:.1f}%
CO2: {last['CO2 (ppm)']:.0f} ppm
CO2 senza rimozione: {last['CO2 senza rimozione (ppm)']:.0f} ppm
Temperatura: +{last['Temperatura (C)']:.2f} C
Overshoot Day: {last['Overshoot Day']}
"""

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Scarica dati completi CSV",
        data=csv_data,
        file_name="simulazione_quantitativa_2075.csv",
        mime="text/csv",
        width="stretch",
    )
with download_col2:
    st.download_button(
        "Scarica sommario scenario",
        data=summary,
        file_name="scenario_summary.txt",
        mime="text/plain",
        width="stretch",
    )

st.caption(
    "Fonti di riferimento: Global Footprint Network per impronta/biocapacita; UN WPP per popolazione; Global Carbon Budget/IPCC per il modulo carbonio-clima."
)
