
# =====================================================================
#  ESTADÍSTICA PRO  -  Versión 1.0
#  Aplicación web de distribuciones muestrales (Ingeniería Industrial)
#
#  Tecnología (todo gratuito):
#    Python + Streamlit (pantalla) + NumPy/Pandas (cálculos)
#    + SciPy (probabilidades) + Matplotlib (gráficas)
#
#  Para abrirla en tu computador:   streamlit run app.py
# =====================================================================

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="ESTADÍSTICA PRO", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------
#  CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------
POBLACION_DEFECTO = "1, 9, 12, 14, 16, 20"
MAX_MUESTRAS = 200_000
AZUL, ROJO, VERDE = "#1f4e8c", "#d9534f", "#2e8b57"

OPCIONES_EVENTO = {
    "Mayor que  (X > a)": "mayor",
    "Menor que  (X < a)": "menor",
    "Entre  (a < X < b)": "entre",
}


# ---------------------------------------------------------------------
#  FUNCIONES DE APOYO
# ---------------------------------------------------------------------
def leer_numeros(texto):
    """Convierte una cadena de números en una lista."""
    partes = texto.replace(";", ",").replace("\n", ",").split(",")
    try:
        return [float(p) for p in partes if p.strip() != ""]
    except ValueError:
        return None


def obtener_poblacion():
    """Devuelve la población guardada en el módulo 1."""
    return st.session_state.get("poblacion", leer_numeros(POBLACION_DEFECTO))


def media_varianza_desviacion(datos):
    """Calcula media, varianza y desviación de una población."""
    d = np.array(datos, dtype=float)
    mu = d.mean()
    varianza = ((d - mu) ** 2).mean()
    sigma = math.sqrt(varianza)
    return mu, varianza, sigma


def media_y_desviacion(datos):
    """Mantiene la función original para los cálculos existentes."""
    mu, varianza, sigma = media_varianza_desviacion(datos)
    return mu, sigma


@st.cache_data
def calcular_muestras(poblacion, n, con_reposicion):
    """Genera todas las muestras posibles de tamaño n y sus medias."""
    if con_reposicion:
        total = len(poblacion) ** n
        generador = itertools.product(poblacion, repeat=n)
    else:
        total = math.comb(len(poblacion), n)
        generador = itertools.combinations(poblacion, n)

    if total > MAX_MUESTRAS:
        return None, total

    filas = [
        {
            "Muestra": "(" + ", ".join(f"{v:g}" for v in m) + ")",
            "Media": float(np.mean(m))
        }
        for m in generador
    ]

    tabla = pd.DataFrame(filas)
    tabla.index = range(1, len(tabla) + 1)
    return tabla, total


def grafica_barras_medias(medias, color, titulo):
    """Gráfica de frecuencias de las medias muestrales."""
    conteo = pd.Series(np.round(medias, 4)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([f"{v:g}" for v in conteo.index], conteo.values, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Media de la muestra")
    ax.set_ylabel("Frecuencia")

    if len(conteo) > 12:
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    fig.tight_layout()
    return fig


def bloque_evento(media, desv, nombre, a0, b0, paso, formato, clave):
    """
    Calcula probabilidades con la distribución normal y muestra
    la media, varianza y desviación estándar.
    """
    if desv <= 0:
        st.error("La desviación debe ser mayor que cero.")
        return

    varianza = desv ** 2

    st.markdown("#### Evento cuya probabilidad quieres calcular")

    eleccion = st.radio(
        "Tipo de evento",
        list(OPCIONES_EVENTO),
        horizontal=True,
        key=f"{clave}_tipo"
    )

    tipo = OPCIONES_EVENTO[eleccion]

    c1, c2 = st.columns(2)

    a = c1.number_input(
        "Valor a",
        value=float(a0),
        step=paso,
        format=formato,
        key=f"{clave}_a"
    )

    b = None

    if tipo == "entre":
        b = c2.number_input(
            "Valor b",
            value=float(b0),
            step=paso,
            format=formato,
            key=f"{clave}_b"
        )

        if b <= a:
            st.warning("El valor b debe ser mayor que a.")
            return

    # Cálculo con la normal
    za = (a - media) / desv

    if tipo == "mayor":
        prob = 1 - stats.norm.cdf(za)
        evento = f"P({nombre} > {a:g})"
        pasos = f"z = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}"

    elif tipo == "menor":
        prob = stats.norm.cdf(za)
        evento = f"P({nombre} < {a:g})"
        pasos = f"z = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}"

    else:
        zb = (b - media) / desv
        prob = stats.norm.cdf(zb) - stats.norm.cdf(za)
        evento = f"P({a:g} < {nombre} < {b:g})"

        pasos = (
            f"z₁ = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}   |   "
            f"z₂ = ({b:g} − {media:.4f}) / {desv:.4f} = {zb:.4f}"
        )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Media esperada", f"{media:.4f}")
    m2.metric("Varianza esperada", f"{varianza:.4f}")
    m3.metric("Desviación esperada", f"{desv:.4f}")
    m4.metric(evento, f"{prob:.4f}")

    st.caption(f"Estandarización: {pasos}")

    # Gráfica de la normal
    x = np.linspace(media - 4 * desv, media + 4 * desv, 600)
    y = stats.norm.pdf(x, media, desv)

    if tipo == "mayor":
        zona = x >= a
    elif tipo == "menor":
        zona = x <= a
    else:
        zona = (x >= a) & (x <= b)

    fig, ax = plt.subplots(figsize=(8, 3.8))

    ax.plot(x, y, color=AZUL, linewidth=2)

    ax.fill_between(
        x[zona],
        y[zona],
        color=ROJO,
        alpha=0.55,
        label=f"{evento} = {prob:.4f}"
    )

    ax.axvline(
        media,
        color="gray",
        linestyle="--",
        label=f"Media = {media:.4f}"
    )

    ax.set_xlabel(nombre)
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Interpretación:** si se repitiera el muestreo muchas veces, "
        f"en aproximadamente el **{prob * 100:.2f} %** de los casos "
        f"se cumpliría el evento {evento}."
    )


# ---------------------------------------------------------------------
#  MÓDULO 1: POBLACIÓN
# ---------------------------------------------------------------------
def modulo_poblacion():
    st.header("1. Población")

    st.write(
        "Escribe los valores de la población separados por comas. "
        "Esta población se usará también en el módulo 2."
    )

    texto = st.text_area(
        "Valores de la población",
        value=st.session_state.get("pob_texto", POBLACION_DEFECTO),
    )

    datos = leer_numeros(texto)

    if datos is None or len(datos) < 2:
        st.error("Escribe al menos 2 números válidos separados por comas.")
        return

    st.session_state["poblacion"] = datos
    st.session_state["pob_texto"] = texto

    N = len(datos)
    mu, varianza, sigma = media_varianza_desviacion(datos)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tamaño de la población (N)", N)
    c2.metric("Media poblacional (μ)", f"{mu:.4f}")
    c3.metric("Varianza poblacional (σ²)", f"{varianza:.4f}")
    c4.metric("Desviación poblacional (σ)", f"{sigma:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.2))

    ax.bar([f"{v:g}" for v in datos], datos, color=AZUL)
    ax.axhline(mu, color=ROJO, linestyle="--", label=f"μ = {mu:.2f}")

    ax.set_xlabel("Valor")
    ax.set_ylabel("Magnitud")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver el cálculo paso a paso"):
        d = np.array(datos)

        detalle = pd.DataFrame({
            "x": d,
            "x − μ": d - mu,
            "(x − μ)²": (d - mu) ** 2
        })

        detalle.index = range(1, N + 1)

        st.dataframe(detalle)

        suma_cuadrados = ((d - mu) ** 2).sum()

        st.write(
            f"μ = Σx / N = {d.sum():g} / {N} = **{mu:.4f}**"
        )

        st.write(
            f"σ² = Σ(x − μ)² / N = {suma_cuadrados:.4f} / {N} "
            f"= **{varianza:.4f}**"
        )

        st.write(
            f"σ = √σ² = √{varianza:.4f} = **{sigma:.4f}**"
        )


# ---------------------------------------------------------------------
#  MÓDULO 2: MEDIAS MUESTRALES
# ---------------------------------------------------------------------
def resumen_muestreo(poblacion, n, con_reposicion, sigma):
    """Calcula la distribución muestral, su media y su varianza."""

    N = len(poblacion)

    if (not con_reposicion) and n > N:
        return None

    tabla, total = calcular_muestras(
        tuple(poblacion),
        n,
        con_reposicion
    )

    if tabla is None:
        return {"tabla": None, "total": total}

    medias = tabla["Media"].values

    mu_x = medias.mean()
    var_x = np.var(medias, ddof=0)
    sig_x = math.sqrt(var_x)

    # Varianza teórica
    var_teorica = sigma ** 2 / n

    if not con_reposicion:
        var_teorica *= (N - n) / (N - 1)

    # Desviación teórica
    teorica = math.sqrt(var_teorica)

    return {
        "tabla": tabla,
        "total": total,
        "mu_x": mu_x,
        "var_x": var_x,
        "sig_x": sig_x,
        "var_teorica": var_teorica,
        "teorica": teorica
    }


def mostrar_muestreo(res, con_reposicion, color):

    if res is None:
        st.warning(
            "Sin reposición, el tamaño de la muestra n no puede ser "
            "mayor que el tamaño de la población N."
        )
        return

    if res["tabla"] is None:
        st.warning(
            f"Hay {res['total']:,} muestras posibles: son demasiadas. "
            "Reduce el tamaño de la muestra n."
        )
        return

    c1, c2, c3 = st.columns(3)

    c1.metric("Muestras posibles", f"{res['total']:,}")
    c2.metric("Media de las medias", f"{res['mu_x']:.4f}")
    c3.metric("Varianza de las medias", f"{res['var_x']:.4f}")

    c4, c5 = st.columns(2)

    c4.metric("Desviación de las medias", f"{res['sig_x']:.4f}")
    c5.metric("Varianza por fórmula", f"{res['var_teorica']:.4f}")

    st.metric("Desviación por fórmula", f"{res['teorica']:.4f}")

    if con_reposicion:
        st.caption(
            "Varianza: σ² / n   |   "
            "Desviación: σ / √n   |   "
            "Número de muestras: Nⁿ"
        )
    else:
        st.caption(
            "Varianza: (σ² / n) · ((N − n) / (N − 1))   |   "
            "Desviación: (σ / √n) · √((N − n) / (N − 1))   |   "
            "Número de muestras: N sobre n"
        )

    titulo = (
        "Medias muestrales "
        + ("CON" if con_reposicion else "SIN")
        + " reposición"
    )

    fig = grafica_barras_medias(
        res["tabla"]["Media"].values,
        color,
        titulo
    )

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver todas las muestras y sus medias"):
        st.dataframe(res["tabla"])


def modulo_medias():
    st.header("2. Medias muestrales")

    poblacion = obtener_poblacion()
    mu, varianza, sigma = media_varianza_desviacion(poblacion)

    st.write(
        f"Población usada: **{', '.join(f'{v:g}' for v in poblacion)}**  "
        f"(μ = {mu:.4f}, σ² = {varianza:.4f}, σ = {sigma:.4f}). "
        "Puedes cambiarla en el módulo 1."
    )

    max_n = min(6, len(poblacion))
    n_inicial = min(2, max_n)

    n = st.slider(
        "Tamaño de la muestra (n)",
        1,
        max_n,
        n_inicial
    )

    con_rep = resumen_muestreo(poblacion, n, True, sigma)
    sin_rep = resumen_muestreo(poblacion, n, False, sigma)

    tab_a, tab_b, tab_c = st.tabs([
        "a) Con reposición",
        "b) Sin reposición",
        "Comparación"
    ])

    with tab_a:
        mostrar_muestreo(con_rep, True, AZUL)

    with tab_b:
        mostrar_muestreo(sin_rep, False, VERDE)

    with tab_c:

        if (
            con_rep
            and sin_rep
            and con_rep["tabla"] is not None
            and sin_rep["tabla"] is not None
        ):

            comp = pd.DataFrame({
                "Con reposición": [
                    con_rep["total"],
                    mu,
                    con_rep["mu_x"],
                    con_rep["var_x"],
                    con_rep["var_teorica"],
                    con_rep["sig_x"],
                    con_rep["teorica"]
                ],
                "Sin reposición": [
                    sin_rep["total"],
                    mu,
                    sin_rep["mu_x"],
                    sin_rep["var_x"],
                    sin_rep["var_teorica"],
                    sin_rep["sig_x"],
                    sin_rep["teorica"]
                ],
            }, index=[
                "Muestras posibles",
                "Media poblacional (μ)",
                "Media de las medias",
                "Varianza de las medias",
                "Varianza teórica",
                "Desviación de las medias",
                "Desviación teórica"
            ])

            st.dataframe(comp.style.format("{:,.4f}"))

            st.info(
                "En ambos casos la media de las medias es igual a μ. "
                "Sin reposición la varianza y la desviación son menores, "
                "porque cada muestra aporta más información de una "
                "población finita."
            )

        else:
            st.write("Ajusta n para poder comparar ambos casos.")


# ---------------------------------------------------------------------
#  MÓDULO 3: PROBABILIDADES
# ---------------------------------------------------------------------
def modulo_probabilidades():
    st.header("3. Probabilidades de medias muestrales")

    st.write(
        "Calcula la probabilidad de que la **media de una muestra** "
        "cumpla un evento, usando la distribución normal."
    )

    fuente = st.radio(
        "¿Qué datos usar?",
        [
            "Mi propio problema (ejemplo de ensamble)",
            "Población del módulo 1"
        ],
        horizontal=True
    )

    if fuente.startswith("Mi propio"):

        st.markdown(
            "**Ejemplo:** una pieza tarda en promedio 45 min en ensamblarse, "
            "con desviación de 8 min. Se toma una muestra de 36 piezas. "
            "¿Cuál es la probabilidad de que el tiempo promedio supere 47 min?"
        )

        c1, c2, c3 = st.columns(3)

        mu = c1.number_input(
            "Media poblacional (μ)",
            value=45.0,
            step=1.0
        )

        sigma = c2.number_input(
            "Desviación poblacional (σ)",
            value=8.0,
            min_value=0.01,
            step=0.5
        )

        n = int(c3.number_input(
            "Tamaño de la muestra (n)",
            value=36,
            min_value=1,
            step=1
        ))

        N = None

        sin_rep = st.checkbox(
            "Muestreo sin reposición en población finita"
        )

        if sin_rep:
            N = int(st.number_input(
                "Tamaño de la población (N)",
                value=max(n + 1, 200),
                min_value=n + 1,
                step=1
            ))

        a0, b0, clave = 47.0, 49.0, "prob_propia"

    else:

        poblacion = obtener_poblacion()
        mu, var_poblacional, sigma = media_varianza_desviacion(poblacion)

        st.write(
            f"Población: **{', '.join(f'{v:g}' for v in poblacion)}**  "
            f"(μ = {mu:.4f}, σ² = {var_poblacional:.4f}, "
            f"σ = {sigma:.4f})"
        )

        max_n = min(6, len(poblacion))

        n = st.slider(
            "Tamaño de la muestra (n)",
            1,
            max_n,
            min(2, max_n),
            key="prob_n_pob"
        )

        sin_rep = st.checkbox("Muestreo sin reposición")
        N = len(poblacion) if sin_rep else None

        if sin_rep and n > N:
            st.warning("Sin reposición, n no puede ser mayor que N.")
            return

        a0, b0, clave = round(mu + 1, 2), round(mu + 3, 2), "prob_pob"

    # Varianza y error estándar de la media muestral
    varianza_error = sigma ** 2 / n

    if sin_rep and N is not None and N > 1:
        varianza_error *= (N - n) / (N - 1)

        st.caption(
            "Varianza del error estándar = (σ²/n) · ((N−n)/(N−1))"
        )

    else:
        st.caption("Varianza del error estándar = σ²/n")

    error = math.sqrt(varianza_error)

    bloque_evento(
        mu,
        error,
        "X̄ (media muestral)",
        a0,
        b0,
        0.5,
        "%.4f",
        clave
    )


# ---------------------------------------------------------------------
#  MÓDULO 4: PROPORCIONES
# ---------------------------------------------------------------------
def modulo_proporciones():
    st.header("4. Distribución muestral de proporciones")

    st.markdown(
        "**Ejemplo:** el 8 % de las piezas sale defectuosa. Se revisa una "
        "muestra de 200 piezas. ¿Cuál es la probabilidad de que la "
        "proporción de defectuosas sea mayor que 0,10?"
    )

    c1, c2 = st.columns(2)

    p = c1.number_input(
        "Proporción poblacional (p)",
        value=0.08,
        min_value=0.001,
        max_value=0.999,
        step=0.01,
        format="%.4f"
    )

    n = int(c2.number_input(
        "Tamaño de la muestra (n)",
        value=200,
        min_value=1,
        step=10
    ))

    if n * p >= 5 and n * (1 - p) >= 5:

        st.success(
            f"Se puede usar la normal: n·p = {n * p:.1f} y "
            f"n·(1−p) = {n * (1 - p):.1f} (ambos ≥ 5)."
        )

    else:

        st.warning(
            f"Ojo: n·p = {n * p:.1f} y n·(1−p) = {n * (1 - p):.1f}. "
            "Si alguno es menor que 5, la aproximación normal no es buena."
        )

    varianza = p * (1 - p) / n
    error = math.sqrt(varianza)

    st.caption(
        "Media de p̂ = p   |   "
        "Varianza de p̂ = p(1−p)/n   |   "
        "Desviación de p̂ = √(p(1−p)/n)"
    )

    bloque_evento(
        p,
        error,
        "p̂ (proporción muestral)",
        round(p + 0.02, 4),
        round(p + 0.04, 4),
        0.01,
        "%.4f",
        "prop"
    )


# ---------------------------------------------------------------------
#  MÓDULO 5: DIFERENCIAS Y SUMAS
# ---------------------------------------------------------------------
def modulo_diferencias_sumas():
    st.header("5. Diferencias y sumas")

    caso = st.selectbox(
        "Elige el caso",
        [
            "Diferencia de medias (dos máquinas)",
            "Diferencia de proporciones (dos turnos)",
            "Suma de variables (tiempo total de pedidos)",
        ]
    )

    if caso.startswith("Diferencia de medias"):

        st.markdown(
            "**Ejemplo:** la máquina 1 tiene tiempo de ciclo medio 50 s "
            "(σ = 6, n = 40) y la máquina 2 tiene 47 s (σ = 5, n = 35). "
            "¿Probabilidad de que la diferencia de medias supere 5 s?"
        )

        c1, c2 = st.columns(2)

        mu1 = c1.number_input("Media máquina 1 (μ₁)", value=50.0)

        s1 = c1.number_input(
            "Desviación máquina 1 (σ₁)",
            value=6.0,
            min_value=0.01
        )

        n1 = int(c1.number_input(
            "Muestra máquina 1 (n₁)",
            value=40,
            min_value=1
        ))

        mu2 = c2.number_input("Media máquina 2 (μ₂)", value=47.0)

        s2 = c2.number_input(
            "Desviación máquina 2 (σ₂)",
            value=5.0,
            min_value=0.01
        )

        n2 = int(c2.number_input(
            "Muestra máquina 2 (n₂)",
            value=35,
            min_value=1
        ))

        media = mu1 - mu2
        varianza = s1 ** 2 / n1 + s2 ** 2 / n2
        desv = math.sqrt(varianza)

        st.caption(
            "Media = μ₁ − μ₂   |   "
            "Varianza = σ₁²/n₁ + σ₂²/n₂   |   "
            "Desviación = √(σ₁²/n₁ + σ₂²/n₂)"
        )

        bloque_evento(
            media, desv, "X̄₁ − X̄₂",
            5.0, 6.0, 0.5, "%.4f", "dif_med"
        )

    elif caso.startswith("Diferencia de proporciones"):

        st.markdown(
            "**Ejemplo:** el turno A produce 10 % de defectuosos (n = 150) "
            "y el turno B produce 7 % (n = 200). ¿Probabilidad de que la "
            "diferencia de proporciones supere 0,05?"
        )

        c1, c2 = st.columns(2)

        pa = c1.number_input(
            "Proporción turno A (p₁)",
            value=0.10,
            min_value=0.001,
            max_value=0.999,
            step=0.01,
            format="%.4f"
        )

        na = int(c1.number_input(
            "Muestra turno A (n₁)",
            value=150,
            min_value=1
        ))

        pb = c2.number_input(
            "Proporción turno B (p₂)",
            value=0.07,
            min_value=0.001,
            max_value=0.999,
            step=0.01,
            format="%.4f"
        )

        nb = int(c2.number_input(
            "Muestra turno B (n₂)",
            value=200,
            min_value=1
        ))

        media = pa - pb

        varianza = (
            pa * (1 - pa) / na
            + pb * (1 - pb) / nb
        )

        desv = math.sqrt(varianza)

        st.caption(
            "Media = p₁ − p₂   |   "
            "Varianza = p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂"
        )

        bloque_evento(
            media, desv, "p̂₁ − p̂₂",
            0.05, 0.08, 0.01, "%.4f", "dif_prop"
        )

    else:

        st.markdown(
            "**Ejemplo:** cada pedido tarda en promedio 10 min (σ = 2). "
            "Se atienden 30 pedidos. ¿Probabilidad de que el tiempo "
            "total supere 320 min?"
        )

        c1, c2, c3 = st.columns(3)

        mu = c1.number_input(
            "Media de cada pedido (μ)",
            value=10.0
        )

        sigma = c2.number_input(
            "Desviación de cada pedido (σ)",
            value=2.0,
            min_value=0.01
        )

        n = int(c3.number_input(
            "Número de pedidos (n)",
            value=30,
            min_value=1
        ))

        media = n * mu
        varianza = n * sigma ** 2
        desv = math.sqrt(varianza)

        st.caption(
            "Media de la suma = n·μ   |   "
            "Varianza de la suma = n·σ²   |   "
            "Desviación de la suma = σ·√n"
        )

        bloque_evento(
            media, desv, "ΣX (suma)",
            320.0, 330.0, 5.0, "%.4f", "suma"
        )


# ---------------------------------------------------------------------
#  MÓDULO 6: GUÍA
# ---------------------------------------------------------------------
def modulo_guia():
    st.header("6. Guía y fórmulas")

    st.subheader("Fórmulas que usa la aplicación")

    st.markdown(r"""
| Tema | Media | Varianza | Desviación |
|---|---|---|---|
| Población | $\mu = \dfrac{\sum x}{N}$ | $\sigma^2 = \dfrac{\sum (x-\mu)^2}{N}$ | $\sigma = \sqrt{\dfrac{\sum (x-\mu)^2}{N}}$ |
| Medias (con reposición) | $\mu_{\bar{x}} = \mu$ | $\sigma_{\bar{x}}^2 = \dfrac{\sigma^2}{n}$ | $\sigma_{\bar{x}} = \dfrac{\sigma}{\sqrt{n}}$ |
| Medias (sin reposición) | $\mu_{\bar{x}} = \mu$ | $\sigma_{\bar{x}}^2 = \dfrac{\sigma^2}{n}\dfrac{N-n}{N-1}$ | $\sigma_{\bar{x}} = \dfrac{\sigma}{\sqrt{n}}\sqrt{\dfrac{N-n}{N-1}}$ |
| Proporciones | $\mu_{\hat{p}} = p$ | $\sigma_{\hat{p}}^2 = \dfrac{p(1-p)}{n}$ | $\sigma_{\hat{p}} = \sqrt{\dfrac{p(1-p)}{n}}$ |
| Diferencia de medias | $\mu_1 - \mu_2$ | $\dfrac{\sigma_1^2}{n_1}+\dfrac{\sigma_2^2}{n_2}$ | $\sqrt{\dfrac{\sigma_1^2}{n_1}+\dfrac{\sigma_2^2}{n_2}}$ |
| Diferencia de proporciones | $p_1 - p_2$ | $\dfrac{p_1(1-p_1)}{n_1}+\dfrac{p_2(1-p_2)}{n_2}$ | $\sqrt{\dfrac{p_1(1-p_1)}{n_1}+\dfrac{p_2(1-p_2)}{n_2}}$ |
| Suma de $n$ variables | $n\mu$ | $n\sigma^2$ | $\sigma\sqrt{n}$ |

Para llevar un valor a la normal estándar:

$z = \dfrac{\text{valor} - \text{media}}{\text{desviación}}$
""")

    st.subheader("¿Qué es la varianza?")

    st.markdown("""
La **varianza** es una medida estadística que indica qué tan dispersos
están los datos con respecto a su media. Se calcula elevando al cuadrado
las diferencias entre cada dato y la media, sumándolas y dividiendo
entre el número de datos cuando se trabaja con una población.

- Una varianza pequeña indica que los datos están más cerca de la media.
- Una varianza grande indica que los datos están más dispersos.
- La desviación estándar es la raíz cuadrada de la varianza.
""")

    st.subheader("¿Por qué el docente pide resolverlo con software?")

    st.markdown("""
1. **Evita errores.** Con 36 muestras, o con 15, el cálculo a mano es largo y fácil de equivocar.
2. **Permite comprobar la teoría.** Ves que la media de las medias es igual a μ y que la varianza y la desviación coinciden con las fórmulas.
3. **Muestra el Teorema del Límite Central.** Al cambiar n, las gráficas se acercan a una campana.
4. **Es una habilidad profesional.** En planta se analizan datos de producción y calidad con herramientas como esta.
5. **Se pueden probar escenarios.** Cambias un dato y todo se recalcula al instante.
""")


# ---------------------------------------------------------------------
#  PANTALLA PRINCIPAL
# ---------------------------------------------------------------------
MODULOS = {
    "1. Población": modulo_poblacion,
    "2. Medias muestrales": modulo_medias,
    "3. Probabilidades": modulo_probabilidades,
    "4. Proporciones": modulo_proporciones,
    "5. Diferencias y sumas": modulo_diferencias_sumas,
    "6. Guía y fórmulas": modulo_guia,
}

st.sidebar.markdown("## 📊 ESTADÍSTICA PRO")
st.sidebar.caption("Versión 1.0 · Ingeniería Industrial")

seleccion = st.sidebar.radio("Módulos", list(MODULOS))

st.title("📊 ESTADÍSTICA PRO")

st.caption(
    "Distribuciones muestrales: población, medias, probabilidades, "
    "proporciones, diferencias y sumas."
)

MODULOS[seleccion]()
