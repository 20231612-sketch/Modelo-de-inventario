import math
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL Y ESTILOS DE STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Suite de Inventario FMCG",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Suite Analítica: Lote Económico, Stock de Seguridad y Punto de Reorden")
st.markdown("""
Esta aplicación construye la política de inventario por categoría a partir de la demanda mensual del caso FMCG.
El costo de ordenar **no se supone**: se deriva de la **curva de intercambio** de Silver, Pyke y Thomas (2017, §4.10).
El nivel de servicio se fija por **fill rate**, resuelto con la **función de pérdida normal estándar** $G(k)$, y el
stock de seguridad se calcula sobre la variabilidad conjunta de la demanda y del tiempo de reposición.
""")

# ---------------------------------------------------------
# ETAPA 1: CARGA DE DATOS
# ---------------------------------------------------------
st.sidebar.header("📁 1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo (CSV o Excel):", type=["csv", "xlsx"])


@st.cache_data
def load_sample_data():
    """Reproduce el dataset del caso (extended_fmcg_demand_forecasting.csv) en su forma diaria.

    Los parámetros de la simulación replican los estadísticos medidos sobre el archivo original:
    Sales_Volume es indistinguible de una Uniforme[101, 1997] (Kolmogorov-Smirnov p = 0,179),
    un registro por día y cinco categorías.
    """
    rng = np.random.default_rng(42)
    fechas = pd.date_range("2022-01-01", periods=1000, freq="D")
    categorias = ["Beverages", "Dairy", "Household", "Personal Care", "Snacks"]
    costo_base = {"Beverages": 7.69, "Dairy": 7.47, "Household": 7.15,
                  "Personal Care": 7.65, "Snacks": 7.81}
    cat = rng.choice(categorias, len(fechas))
    return pd.DataFrame({
        "Date": fechas,
        "Product_Category": cat,
        "Sales_Volume": rng.integers(101, 1998, len(fechas)),
        "Price": rng.uniform(1.0, 19.95, len(fechas)).round(4),
        "Promotion": rng.integers(0, 2, len(fechas)),
        "Store_Location": rng.choice(["Urban", "Suburban", "Rural"], len(fechas)),
        "Weekday": [d.weekday() for d in fechas],
        "Supplier_Cost": [max(0.5, rng.normal(costo_base[c], 4.16)) for c in cat],
        "Replenishment_Lead_Time": rng.integers(1, 10, len(fechas)),
        "Stock_Level": rng.integers(50, 500, len(fechas)),
    })


DATA_REPO = Path(__file__).parent / "data" / "extended_fmcg_demand_forecasting.csv"


@st.cache_data
def load_repo_data():
    """Dataset del caso incluido en el repositorio (CSV separado por ';' o ',': se detecta solo)."""
    return pd.read_csv(DATA_REPO, sep=None, engine="python")


if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("¡Archivo cargado con éxito!")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()
elif DATA_REPO.exists():
    df = load_repo_data().copy()
    st.sidebar.info("Usando el dataset del caso incluido en el repositorio "
                    "(`extended_fmcg_demand_forecasting.csv`). Puedes subir otro archivo para reemplazarlo.")
else:
    df = load_sample_data()
    st.sidebar.info("Cargado dataset de ejemplo equivalente al del caso. Sube "
                    "`extended_fmcg_demand_forecasting.csv` para trabajar con los datos reales.")

# Normalización de la columna de fecha (el archivo del caso viene en formato día/mes/año)
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

with st.expander("👀 Vista Previa y Resumen del Dataset", expanded=True):
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(df.head(8), width="stretch")
    with col_b:
        st.write(f"**Filas:** {df.shape[0]}")
        st.write(f"**Columnas:** {df.shape[1]}")
        if "Date" in df.columns and df["Date"].notna().any():
            st.write(
                f"**Rango:** {df['Date'].min():%d/%m/%Y} — {df['Date'].max():%d/%m/%Y}")
        st.write("**Tipos de datos detectados:**")
        st.write(df.dtypes.astype(str).to_frame(name="Tipo"))

# ---------------------------------------------------------
# ETAPA 2: CONSTRUCCIÓN DE LA DEMANDA MENSUAL
# ---------------------------------------------------------
st.header("⚙️ 2. Construcción de la Demanda Mensual y Parámetros por Categoría")

col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    col_cat = st.selectbox("Columna de categoría:",
                           options=df.columns.tolist(),
                           index=df.columns.tolist().index("Product_Category")
                           if "Product_Category" in df.columns else 0)
with col_v2:
    col_dem = st.selectbox("Columna de demanda (Y):",
                           options=df.select_dtypes(
                               include=[np.number]).columns.tolist(),
                           index=df.select_dtypes(
                               include=[np.number]).columns.tolist().index("Sales_Volume")
                           if "Sales_Volume" in df.select_dtypes(include=[np.number]).columns.tolist() else 0)
with col_v3:
    num_opts = df.select_dtypes(include=[np.number]).columns.tolist()
    col_cost = st.selectbox("Columna de costo unitario (v):",
                            options=num_opts,
                            index=num_opts.index("Supplier_Cost") if "Supplier_Cost" in num_opts else 0)

escenario = st.radio(
    "Forma de construir la demanda mensual (unidad de análisis: **categoría × mes**):",
    ["A · Suma cruda de los registros del mes",
     "B · Media diaria × días del mes",
     "C · σ diario × √días  (estimador eficiente)"],
    help="La elección determina tanto el nivel como la desviación que alimentan el stock de seguridad."
)

NOTAS = {
    "A": ("El total mensual correlaciona **0,889** con el número de registros de la celda (R² = 0,79) y solo "
          "**0,283** con el nivel medio de venta. Como el dataset contiene un único registro por día, cada celda "
          "categoría-mes reúne ~6 días de los ~30 del mes: cerca del 79 % de la varianza contra la que protege el "
          "colchón es artefacto de muestreo, y la escala queda subestimada unas 5 veces."),
    "B": ("Corrige la escala y elimina el artefacto de conteo, pero introduce un sesgo propio: la media mensual se "
          "estima con ~6,06 días en lugar de 30,4, lo que infla σ por un factor observado de **2,36** "
          "(teórico √(30,4/6,06) = 2,24). Supone que cada categoría se vende los 30 días del mes."),
    "C": ("Estima la variabilidad con ~200 observaciones diarias por categoría y la escala por √30,4 bajo "
          "independencia entre días. Es el estimador más eficiente y lleva el c.v. por debajo del umbral de 0,40, "
          "pero deja la demanda tan estable que el stock de seguridad pierde relevancia. Comparte con B el supuesto "
          "de extrapolación a 30 días.")
}
clave_esc = escenario[0]
st.info(f"💡 **Escenario {clave_esc}.** {NOTAS[clave_esc]}")

st.sidebar.header("🎛️ 2. Parámetros Globales")
lt_dias = st.sidebar.number_input(
    "Lead time medio (días):", 0.1, 60.0, 4.97, 0.01)
slt_dias = st.sidebar.number_input(
    "Desviación del lead time (días):", 0.0, 30.0, 2.59, 0.01)
tasa_r = st.sidebar.slider("Tasa de mantener r (% anual):", 5, 40, 20,
                           help="Gurtu (2021): la literatura sitúa r entre 16 % y 20 % anual del costo del ítem.")
fill_rate = st.sidebar.slider("Fill rate objetivo (%):", 80.0, 99.9, 97.0, 0.5)
n_repos = st.sidebar.number_input("Reposiciones totales del sistema (N/año):", 10, 1000, 260, 10,
                                  help="Decisión de política que fija el costo de ordenar vía curva de intercambio.")
dias_mes = st.sidebar.number_input("Días por mes:", 28.0, 31.0, 30.4, 0.1)


@st.cache_data
def construir_parametros(datos, c_cat, c_dem, c_cost, modo, dpm):
    """Devuelve d̄ (demanda mensual), σ y v por categoría según el escenario elegido."""
    d = datos[[c_cat, c_dem, c_cost]].dropna().copy()
    if "Date" in datos.columns:
        d["Date"] = datos.loc[d.index, "Date"]
        d["YM"] = d["Date"].dt.to_period("M")

    filas = []
    for categoria, sub in d.groupby(c_cat):
        v = float(sub[c_cost].mean())
        if modo == "A" and "YM" in sub.columns:
            mensual = sub.groupby("YM")[c_dem].sum()
            d_bar, sigma = float(mensual.mean()), float(mensual.std(ddof=1))
        elif modo == "B" and "YM" in sub.columns:
            mensual = sub.groupby("YM")[c_dem].mean() * dpm
            d_bar, sigma = float(mensual.mean()), float(mensual.std(ddof=1))
        else:  # C — estimación diaria escalada
            d_bar = float(sub[c_dem].mean()) * dpm
            sigma = float(sub[c_dem].std(ddof=1)) * math.sqrt(dpm)
        filas.append({"Categoría": categoria, "d̄ mensual": d_bar, "σ mensual": sigma,
                      "v unitario": v, "c.v.": sigma / d_bar if d_bar else np.nan,
                      "D anual": d_bar * 12})
    return pd.DataFrame(filas).sort_values("Categoría").reset_index(drop=True)


par = construir_parametros(df, col_cat, col_dem, col_cost, clave_esc, dias_mes)

if par.empty or len(par) < 2:
    st.error(
        "No se pudieron construir parámetros por categoría. Revisa las columnas seleccionadas.")
    st.stop()

st.dataframe(
    par.style.format({"d̄ mensual": "{:,.0f}", "σ mensual": "{:,.0f}", "v unitario": "{:,.2f}",
                      "c.v.": "{:.3f}", "D anual": "{:,.0f}"}),
    width="stretch"
)

altos = par[par["c.v."] > 0.40]
if len(altos) > 0:
    st.warning(f"⚠️ **Coeficiente de variación por encima del umbral:** {len(altos)} de {len(par)} categorías superan "
               "un c.v. de 0,40. Tyworth y O'Neill (1997), citados por Chopra, Reinhardt y Dada (2004), sitúan ahí el "
               "límite de confiabilidad de la aproximación normal. Los resultados siguen siendo válidos como orden de "
               "magnitud, pero la limitación debe declararse en el informe.")
else:
    st.success("✅ **Coeficiente de variación dentro del rango confiable:** todas las categorías están por debajo de "
               "0,40, umbral bajo el cual la aproximación normal se considera confiable.")

# ---------------------------------------------------------
# ETAPA 3: COSTO DE ORDENAR DERIVADO (CURVA DE INTERCAMBIO)
# ---------------------------------------------------------
st.header("🧮 3. Costo de Ordenar Derivado de la Curva de Intercambio")

st.markdown("""
Silver, Pyke y Thomas (2017, §4.10) plantean el método porque *«often, it is difficult to explicitly determine an
appropriate value of the carrying charge r or the fixed cost per replenishment A»*. Fijado el número de reposiciones
al año, la curva implica un cociente $A/r$; conocida $r$, queda $A$ **derivado** y no supuesto.
""")

st.latex(r"TACS \cdot N = \tfrac{1}{2}\left(\sum_i \sqrt{D_i v_i}\right)^2 \qquad "
         r"\frac{TACS}{N} = \frac{A}{r} \qquad \Rightarrow \qquad "
         r"A = \frac{r\left(\sum_i \sqrt{D_i v_i}\right)^2}{2N^2}")

r = tasa_r / 100.0
fr = fill_rate / 100.0
LT = lt_dias / dias_mes
sLT = slt_dias / dias_mes

suma_raiz = float(np.sum(np.sqrt(par["D anual"] * par["v unitario"])))
ratio_Ar = suma_raiz ** 2 / (2 * n_repos ** 2)
A = r * ratio_Ar
TACS = 0.5 * suma_raiz ** 2 / n_repos

col_a1, col_a2, col_a3, col_a4 = st.columns(4)
col_a1.metric(r"$\sum \sqrt{D_i v_i}$", f"{suma_raiz:,.1f}")
col_a2.metric("A / r implícito", f"{ratio_Ar:,.2f}")
col_a3.metric("A derivado", f"{A:,.2f}", help="Unidades monetarias por pedido")
col_a4.metric("Stock de ciclo (TACS)", f"{TACS:,.0f}")

# ---------------------------------------------------------
# ETAPA 4: POLÍTICA DE INVENTARIO
# ---------------------------------------------------------
st.header("📈 4. Política de Inventario por Categoría")


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def loss_G(k):
    """Función de pérdida normal estándar: G(k) = φ(k) − k·[1 − Φ(k)]."""
    return norm_pdf(k) - k * (1 - norm_cdf(k))


def solve_k(objetivo):
    """Resuelve G(k) = objetivo por bisección. G es monótona decreciente en k."""
    if not np.isfinite(objetivo) or objetivo <= 0:
        return 8.0
    lo, hi = -6.0, 8.0
    if objetivo >= loss_G(lo):
        return lo
    for _ in range(200):
        mid = (lo + hi) / 2
        if loss_G(mid) > objetivo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calcular_politica(parametros, A_val, r_val, fr_val, LT_val, sLT_val):
    out = []
    for _, fila in parametros.iterrows():
        d_bar, sigma, v, D = fila["d̄ mensual"], fila["σ mensual"], fila["v unitario"], fila["D anual"]
        term_dem = LT_val * sigma ** 2
        term_lt = d_bar ** 2 * sLT_val ** 2
        sDL = math.sqrt(term_dem + term_lt)
        Q = math.sqrt(2 * A_val * D / (r_val * v))
        k = solve_k(Q * (1 - fr_val) / sDL)
        SS = k * sDL
        ROP = d_bar * LT_val + SS
        inv_medio = Q / 2 + SS
        costo = (D / Q) * A_val + inv_medio * r_val * v
        out.append({
            "Categoría": fila["Categoría"], "σ durante LT": sDL, "EOQ (Q*)": Q, "k": k,
            "Stock seguridad": SS, "Punto de reorden": ROP, "CSL equivalente": norm_cdf(k),
            "Rotación": D / inv_medio if inv_medio > 0 else np.nan, "Costo anual": costo,
            "_term_dem": term_dem, "_term_lt": term_lt, "_inv": inv_medio, "_D": D
        })
    return pd.DataFrame(out)


res = calcular_politica(par, A, r, fr, LT, sLT)

st.dataframe(
    res.drop(columns=["_term_dem", "_term_lt", "_inv", "_D"]).style.format({
        "σ durante LT": "{:,.0f}", "EOQ (Q*)": "{:,.0f}", "k": "{:.3f}", "Stock seguridad": "{:,.0f}",
        "Punto de reorden": "{:,.0f}", "CSL equivalente": "{:.1%}", "Rotación": "{:.1f}", "Costo anual": "{:,.0f}"
    }),
    width="stretch"
)

D_total = res["_D"].sum()
inv_total = res["_inv"].sum()
rotacion = D_total / inv_total if inv_total > 0 else np.nan
dias_cobertura = 365 / \
    rotacion if rotacion and np.isfinite(rotacion) else np.nan

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Stock de Seguridad Total",
              f"{res['Stock seguridad'].sum():,.0f}", help="Unidades, todas las categorías")
col_k2.metric("Costo Total Anual",
              f"{res['Costo anual'].sum():,.0f}", help="Ordenar + mantener")
col_k3.metric("Rotación del Sistema", f"{rotacion:,.1f}")
col_k4.metric("Días de Inventario", f"{dias_cobertura:,.1f}")

if res["Stock seguridad"].min() < 0:
    st.error("🚩 **Stock de seguridad negativo.** Cuando el lote es muy grande frente a σ durante el lead time, el "
             "objetivo $G(k)=Q(1-\\text{fill rate})/\\sigma_{DL}$ supera $G(0)=0{,}3989$ y $k$ se vuelve negativo. "
             "Es matemáticamente consistente —un lote enorme alcanza el fill rate sin colchón— pero señala que el "
             "punto de operación quedó fuera del rango sensato: reduce N o revisa la construcción de la demanda.")

if np.isfinite(rotacion) and not (12 <= rotacion <= 20):
    st.warning(f"⚠️ **Rotación fuera de la referencia sectorial.** El modelo arroja {rotacion:,.1f} vueltas al año, "
               "mientras que la referencia habitual en consumo masivo se sitúa entre 12 y 20. Conviene calibrar el "
               "número de reposiciones N contra esa referencia (ver el gráfico de la etapa 6).")

csl_prom = res["CSL equivalente"].mean()
st.info(f"💡 **Fill rate frente a nivel de servicio de ciclo.** Con un fill rate de {fill_rate:.1f} %, el CSL "
        f"equivalente promedio ronda **{csl_prom:.1%}**. Son métricas distintas y confundirlas infla el inventario; "
        "la brecha es la que documentan Chopra, Reinhardt y Dada (2004), quienes reportan que muchas empresas operan "
        "con CSL de 50–70 % mientras creen estar en 95–99 %.")

# ---------------------------------------------------------
# ETAPA 5: EXPLICACIÓN MATEMÁTICA
# ---------------------------------------------------------
st.header("📐 5. Explicación Matemática del Modelo")

with st.expander("📖 Ver las ecuaciones que resuelve la aplicación", expanded=True):
    st.markdown("**a) Variabilidad de la demanda durante el tiempo de reposición** "
                "(identidad de Wald y ley de la varianza total):")
    st.latex(
        r"\sigma_{DL} = \sqrt{\,LT \cdot \sigma_d^{\,2} \;+\; \mu_d^{\,2}\cdot \sigma_{LT}^{\,2}\,}")

    st.markdown(
        "**b) Lote económico** (minimización convexa del costo total, Harris 1913):")
    st.latex(r"Q^{*} = \sqrt{\frac{2AD}{r\,v}}")

    st.markdown(
        "**c) Factor de seguridad por fill rate**, resuelto con la función de pérdida normal estándar:")
    st.latex(r"\text{fill rate} = 1 - \frac{\sigma_{DL}\,G(k)}{Q^{*}}, \qquad "
             r"G(k) = \varphi(k) - k\,\bigl[1-\Phi(k)\bigr]")

    st.markdown("**d) Stock de seguridad y punto de reorden:**")
    st.latex(r"SS = k\,\sigma_{DL} \qquad\qquad ROP = \mu_d \cdot LT + SS")

    st.caption("El lote Q interviene en la determinación de k, pero no a la inversa: Q proviene del lote económico, "
               "calculado con independencia del nivel de servicio. La dependencia es unidireccional y el sistema se "
               "resuelve de forma secuencial, sin iteración.")

# ---------------------------------------------------------
# ETAPA 6: DIAGNÓSTICO GRÁFICO
# ---------------------------------------------------------
st.header("🔍 6. Diagnóstico Gráfico")

sns.set_style("whitegrid")

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.subheader("Composición del Inventario por Categoría")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(res))
    ax1.bar(idx, res["EOQ (Q*)"] / 2,
            label="Stock de ciclo (Q/2)", color="#1f77b4")
    ax1.bar(idx, res["Stock seguridad"], bottom=res["EOQ (Q*)"] / 2,
            label="Stock de seguridad", color="#ff7f0e")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(res["Categoría"], rotation=20, ha="right")
    ax1.set_ylabel("Unidades")
    ax1.set_title("Inventario medio: ciclo frente a seguridad")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig1)

with fig_col2:
    st.subheader("Origen de la Varianza durante el Lead Time")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    tot_var = res["_term_dem"] + res["_term_lt"]
    pct_dem = res["_term_dem"] / tot_var * 100
    pct_lt = res["_term_lt"] / tot_var * 100
    ax2.barh(res["Categoría"], pct_dem,
             label="Variabilidad de la demanda", color="#1f77b4")
    ax2.barh(res["Categoría"], pct_lt, left=pct_dem,
             label="Variabilidad del lead time", color="#2ca02c")
    ax2.set_xlabel("% de $\\sigma^2_{DL}$")
    ax2.set_xlim(0, 100)
    ax2.set_title("¿Dónde está realmente la incertidumbre?")
    ax2.legend(loc="upper center", bbox_to_anchor=(
        0.5, -0.14), ncol=2, frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    st.pyplot(fig2)

st.subheader("Sensibilidad al Fill Rate Objetivo")
niveles = [90, 93, 95, 97, 99, 99.5]
sens = []
for f in niveles:
    rr = calcular_politica(par, A, r, f / 100.0, LT, sLT)
    sens.append({"Fill rate (%)": f, "Stock de seguridad": rr["Stock seguridad"].sum(),
                 "CSL equivalente (%)": rr["CSL equivalente"].mean() * 100,
                 "Costo anual": rr["Costo anual"].sum()})
sens = pd.DataFrame(sens)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 3.8))
ax3a.plot(sens["Fill rate (%)"], sens["Stock de seguridad"],
          marker="o", color="#ff7f0e", lw=2)
ax3a.axvline(fill_rate, color="black", linestyle="--",
             alpha=0.6, label="Objetivo actual")
ax3a.set_xlabel("Fill rate objetivo (%)")
ax3a.set_ylabel("Stock de seguridad (unidades)")
ax3a.set_title("Costo en inventario de cada punto de servicio")
ax3a.legend()
ax3a.grid(True, linestyle="--", alpha=0.5)

ax3b.plot(sens["Fill rate (%)"], sens["CSL equivalente (%)"], marker="s", color="#1f77b4", lw=2,
          label="CSL equivalente")
ax3b.plot(sens["Fill rate (%)"], sens["Fill rate (%)"], color="gray", linestyle=":", lw=1.5,
          label="Identidad (fill rate = CSL)")
ax3b.set_xlabel("Fill rate objetivo (%)")
ax3b.set_ylabel("Nivel de servicio de ciclo (%)")
ax3b.set_title("La brecha que documentan Chopra et al. (2004)")
ax3b.legend()
ax3b.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig3)

st.dataframe(sens.style.format({"Fill rate (%)": "{:.1f}", "Stock de seguridad": "{:,.0f}",
                                "CSL equivalente (%)": "{:.1f}", "Costo anual": "{:,.0f}"}),
             width="stretch")

st.subheader("Calibración del Número de Reposiciones contra la Rotación")
enes = [20, 40, 60, 80, 104, 156, 208, 260, 400]
calib = []
for N in enes:
    A_n = r * suma_raiz ** 2 / (2 * N ** 2)
    rr = calcular_politica(par, A_n, r, fr, LT, sLT)
    inv_n = rr["_inv"].sum()
    calib.append({"N (reposiciones/año)": N, "A derivado": A_n,
                  "Stock de seguridad": rr["Stock seguridad"].sum(),
                  "Rotación": rr["_D"].sum() / inv_n if inv_n > 0 else np.nan,
                  "Costo anual": rr["Costo anual"].sum()})
calib = pd.DataFrame(calib)

fig4, ax4 = plt.subplots(figsize=(12, 3.5))
ax4.plot(calib["N (reposiciones/año)"], calib["Rotación"],
         marker="o", color="#1f77b4", lw=2)
ax4.axhspan(12, 20, color="#2ca02c", alpha=0.15,
            label="Referencia FMCG (12–20 vueltas)")
ax4.axvline(n_repos, color="black", linestyle="--",
            alpha=0.6, label="N actual")
ax4.set_xlabel("Reposiciones totales del sistema por año (N)")
ax4.set_ylabel("Rotación de inventario")
ax4.set_title("Elección de N y rotación resultante")
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig4)

st.dataframe(calib.style.format({"A derivado": "{:,.2f}", "Stock de seguridad": "{:,.0f}",
                                 "Rotación": "{:.1f}", "Costo anual": "{:,.0f}"}),
             width="stretch")

# ---------------------------------------------------------
# ETAPA 7: SUPUESTOS Y LIMITACIONES
# ---------------------------------------------------------
st.header("📋 7. Supuestos y Limitaciones a Declarar")

with st.expander("📝 Ver el detalle metodológico", expanded=False):
    st.markdown("""
**Unidad de análisis.** Categoría × mes, 33 meses sin celdas vacías. Se descartó categoría × semana porque el 21,9 %
de las celdas semana-categoría quedaban vacías.

**Conversión del lead time.** Se expresa en meses (días ÷ días por mes) para conservar los estadísticos mensuales
medidos, en lugar de convertir la demanda a diaria. Introduce menos supuestos, pero debe declararse.

**La desviación que corresponde usar.** El stock de seguridad protege contra el error de predicción, no contra la
variabilidad ya explicada por el modelo. El pronóstico adoptado en la dimensión 1 es el de la media, y para una serie
sin estructura temporal su error cumple RMSE = σ·√(1+1/n) ≈ σ: el RMSE del pronóstico y la desviación histórica
coinciden, de modo que el valor cargado es el correcto.

**Por qué no se usa Holt-Winters.** Sobre estas series el suavizamiento exponencial triple pierde contra pronósticos
triviales (Theil U de 1,56 con parámetros estándar y 1,41 optimizados, frente al pronóstico por la media). Aislando
componentes, la varianza del error pasa de 2,21× la varianza del dato con estacionalidad activa a 0,98× sin ella: el
componente estacional inyecta ruido en lugar de extraer señal, al estimar doce índices —cada uno desde una sola
observación— sobre 33 datos.

**Aproximación normal.** La fórmula de σ durante el lead time supone demanda normal. Chopra, Reinhardt y Dada (2004)
demuestran que sus prescripciones se distorsionan cuando el coeficiente de variación es alto, y citan a Tyworth y
O'Neill (1997), para quienes la aproximación es confiable con c.v. por debajo de 40 %.

**Costo de mantener uniforme.** Se aplica la misma tasa r a todas las categorías. Gurtu (2021) demuestra que el costo
de mantener varía por ítem según precio, peso y volumen, y que diferenciarlo produjo cerca de 3 % de ahorro.
Limitación reconocida.

**Fuera de alcance.** No se modela costo de venta perdida ni el trade-off de Theodorou et al. (2025). Tampoco
descuentos por volumen ni restricciones de capacidad.
""")

st.caption("Curso Modelamiento Predictivo de Datos · Caso FMCG (Kaggle) · Dimensión 2: Inventario · "
           "Universidad de Lima")
