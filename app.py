import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Simulador Campo Elétrico Fio",
    page_icon="⚡",
    layout="wide",
)

EPS0 = 8.8541878128e-12  # F/m
K = 1 / (4 * math.pi * EPS0)

# ============================================================
# ESTILOS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .scroll-hint {
        font-size: 0.92rem;
        color: #6b7280;
        margin-top: -0.25rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================
def sci_parts(value: float):
    """Retorna mantissa e expoente para notação científica base 10."""
    if value == 0 or abs(value) < 1e-300:
        return 0.0, 0
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / (10 ** exp)
    return mant, exp


def superscript(n: int) -> str:
    """Converte inteiro para sobrescrito Unicode."""
    trans = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(n).translate(trans)


def fmt_num(value: float, digits: int = 4, unit: str | None = None) -> str:
    """
    Formata número para texto comum.
    Usa 10^n no estilo visual com sobrescrito quando necessário.
    """
    if value is None:
        return "—"

    if abs(value) == 0:
        s = "0"
    elif abs(value) >= 1e4 or abs(value) < 1e-3:
        mant, exp = sci_parts(value)
        s = f"{mant:.{digits}f}×10{superscript(exp)}"
    else:
        s = f"{value:.{digits}f}".rstrip("0").rstrip(".")

    return f"{s} {unit}" if unit else s


def fmt_num_plain(value: float, digits: int = 6) -> str:
    """
    Formata número para uso em LaTeX, usando \\times 10^{n} quando necessário.
    """
    if value is None:
        return "—"

    if abs(value) == 0:
        return "0"

    if abs(value) >= 1e4 or abs(value) < 1e-3:
        mant, exp = sci_parts(value)
        return f"{mant:.{digits}f}\\times 10^{{{exp}}}"

    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def theta_from_x_a(x: float, a: float) -> float:
    """
    theta = arctan(x / a), em rad.
    Usa atan2 para robustez numérica.
    """
    return math.atan2(x, a)


def sign_color(lmbd: float) -> str:
    """Cor do fio conforme o sinal da densidade linear."""
    if lmbd > 0:
        return "#d62728"  # vermelho
    if lmbd < 0:
        return "#1f77b4"  # azul
    return "#111111"      # preto


def maybe_logo(path: str = "logo_maua.png"):
    """Carrega logo se existir."""
    p = Path(path)
    if p.exists():
        return Image.open(p)
    return None


def calcular_campo(lambda_lin: float, a: float, x1: float | None, x2: float | None):
    """
    x1, x2 são distâncias POSITIVAS.
    None representa infinito:
      - x1 = None  -> x1 -> infinito (na figura mostraremos -∞, à esquerda)
      - x2 = None  -> x2 -> infinito (na figura mostraremos +∞, à direita)
    """

    theta1 = math.pi / 2 if x1 is None else theta_from_x_a(x1, a)
    theta2 = math.pi / 2 if x2 is None else theta_from_x_a(x2, a)

    inv_r1 = 0.0 if x1 is None else 1 / math.sqrt(x1**2 + a**2)
    inv_r2 = 0.0 if x2 is None else 1 / math.sqrt(x2**2 + a**2)

    s1 = 1.0 if x1 is None else x1 / math.sqrt(x1**2 + a**2)
    s2 = 1.0 if x2 is None else x2 / math.sqrt(x2**2 + a**2)

    # Fórmulas em função das distâncias
    Ex_dist = K * lambda_lin * (inv_r2 - inv_r1)
    Ey_dist = K * lambda_lin * (1 / a) * (s2 + s1)

    # Fórmulas em função dos ângulos
    Ex_ang = K * lambda_lin * (1 / a) * (math.cos(theta2) - math.cos(theta1))
    Ey_ang = K * lambda_lin * (1 / a) * (math.sin(theta1) + math.sin(theta2))

    E = math.sqrt(Ex_dist**2 + Ey_dist**2)
    phi_deg = math.degrees(math.atan2(Ey_dist, Ex_dist))

    return {
        "theta1_rad": theta1,
        "theta2_rad": theta2,
        "theta1_deg": math.degrees(theta1),
        "theta2_deg": math.degrees(theta2),
        "inv_r1": inv_r1,
        "inv_r2": inv_r2,
        "s1": s1,
        "s2": s2,
        "Ex_dist": Ex_dist,
        "Ey_dist": Ey_dist,
        "Ex_ang": Ex_ang,
        "Ey_ang": Ey_ang,
        "E": E,
        "phi_deg": phi_deg,
    }


def finite_or_inf_ui(label: str, key_prefix: str, default_value: float):
    """
    UI para comprimento com slider + modo finito/infinito.
    """
    c1, c2 = st.columns([2.3, 1.2])

    val = c1.slider(
        label,
        min_value=0.0,
        max_value=20.0,
        value=float(default_value),
        step=0.1,
        key=f"slider_{key_prefix}",
    )

    modo = c2.selectbox(
        f"Modo {label}",
        options=["Finito", "Infinito"],
        index=0,
        key=f"mode_{key_prefix}",
    )

    if modo == "Infinito":
        if key_prefix == "x1":
            st.caption("Usando x₁ = −∞ (fio vai infinitamente para a esquerda).")
        else:
            st.caption("Usando x₂ = +∞ (fio vai infinitamente para a direita).")
        return None

    return float(val)


# ============================================================
# FIGURA (PLOTLY)
# ============================================================
def build_plot(lambda_lin: float, a: float, x1: float | None, x2: float | None, calc: dict):
    wire_color = sign_color(lambda_lin)

    finite_values = [v for v in [x1, x2, a] if v is not None]
    base = max(finite_values) if finite_values else a
    span = max(4.0, 1.5 * base)

    # Extensões visuais do fio
    left_vis = -(x1 if x1 is not None else span)
    right_vis = x2 if x2 is not None else span

    # Limites da figura para não cortar elementos
    left_lim = min(left_vis - 0.30 * span, -0.95 * span)
    right_lim = max(right_vis + 0.30 * span, 0.95 * span)
    top_lim = max(a + 0.75 * span, 1.45 * a)
    bottom_lim = min(-0.58 * span, -0.50 * a)

    fig = go.Figure()

    # ---------------------------
    # Fio
    # ---------------------------
    fig.add_trace(
        go.Scatter(
            x=[left_vis, right_vis],
            y=[0, 0],
            mode="lines",
            line=dict(color=wire_color, width=8),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Continuidade visual para fio infinito
    if x1 is None:
        fig.add_annotation(
            x=left_vis,
            y=0,
            ax=left_vis + 0.25 * span,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor=wire_color,
            text="",
        )

    if x2 is None:
        fig.add_annotation(
            x=right_vis,
            y=0,
            ax=right_vis - 0.25 * span,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor=wire_color,
            text="",
        )

    # ---------------------------
    # Ponto P
    # ---------------------------
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[a],
            mode="markers+text",
            marker=dict(size=12, color="#111111"),
            text=["P"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Linha vertical de referência
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, a],
            mode="lines",
            line=dict(color="#666666", width=2, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Diagonais até as extremidades visíveis
    xl = left_vis
    xr = right_vis

    fig.add_trace(
        go.Scatter(
            x=[0, xl],
            y=[a, 0],
            mode="lines",
            line=dict(color="#888888", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, xr],
            y=[a, 0],
            mode="lines",
            line=dict(color="#888888", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ---------------------------
    # COTAS x1 e x2
    # ---------------------------
    y_dim = max(bottom_lim + 0.20 * (top_lim - bottom_lim), -0.22 * span)

    # Linhas de apoio
    for x in [xl, 0, xr]:
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, y_dim],
                mode="lines",
                line=dict(color="#999999", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # x1 (duas setas opostas)
    fig.add_annotation(
        x=xl, y=y_dim, ax=0, ay=y_dim,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )
    fig.add_annotation(
        x=0, y=y_dim, ax=xl, ay=y_dim,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )

    txt_x1 = "x₁ = −∞" if x1 is None else f"x₁ = {fmt_num(x1, 3, 'm')}"
    fig.add_annotation(
        x=(xl + 0) / 2,
        y=y_dim - 0.06 * span,
        text=txt_x1,
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.78)",
    )

    # x2
    fig.add_annotation(
        x=0, y=y_dim, ax=xr, ay=y_dim,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )
    fig.add_annotation(
        x=xr, y=y_dim, ax=0, ay=y_dim,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )

    txt_x2 = "x₂ = +∞" if x2 is None else f"x₂ = {fmt_num(x2, 3, 'm')}"
    fig.add_annotation(
        x=(0 + xr) / 2,
        y=y_dim - 0.06 * span,
        text=txt_x2,
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.78)",
    )

    # ---------------------------
    # COTA a
    # ---------------------------
    x_dim_a = right_lim - 0.08 * (right_lim - left_lim)

    fig.add_trace(
        go.Scatter(
            x=[0, x_dim_a],
            y=[0, 0],
            mode="lines",
            line=dict(color="#999999", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, x_dim_a],
            y=[a, a],
            mode="lines",
            line=dict(color="#999999", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=x_dim_a, y=a, ax=x_dim_a, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )
    fig.add_annotation(
        x=x_dim_a, y=0, ax=x_dim_a, ay=a,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.8, arrowcolor="#555", text=""
    )

    fig.add_annotation(
        x=x_dim_a + 0.08 * span,
        y=a / 2,
        text=f"a = {fmt_num(a, 3, 'm')}",
        textangle=-90,
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.78)",
    )

    # ---------------------------
    # ARCOS DOS ÂNGULOS
    # ---------------------------
    r_arc = 0.22 * span
    t1 = calc["theta1_rad"]
    t2 = calc["theta2_rad"]

    # Ângulo esquerdo: da vertical para a diagonal da esquerda
    phi1 = np.linspace(-math.pi / 2 - t1, -math.pi / 2, 60)
    fig.add_trace(
        go.Scatter(
            x=r_arc * np.cos(phi1),
            y=a + r_arc * np.sin(phi1),
            mode="lines",
            line=dict(color="#9467bd", width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Ângulo direito: da vertical para a diagonal da direita
    phi2 = np.linspace(-math.pi / 2, -math.pi / 2 + t2, 60)
    fig.add_trace(
        go.Scatter(
            x=r_arc * np.cos(phi2),
            y=a + r_arc * np.sin(phi2),
            mode="lines",
            line=dict(color="#2ca02c", width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0.62 * r_arc * math.cos(-math.pi / 2 - t1 / 2),
        y=a + 0.62 * r_arc * math.sin(-math.pi / 2 - t1 / 2),
        text="θ₁",
        showarrow=False,
        font=dict(size=16, color="#6f42c1"),
        bgcolor="rgba(255,255,255,0.78)",
    )

    fig.add_annotation(
        x=0.62 * r_arc * math.cos(-math.pi / 2 + t2 / 2),
        y=a + 0.62 * r_arc * math.sin(-math.pi / 2 + t2 / 2),
        text="θ₂",
        showarrow=False,
        font=dict(size=16, color="#22863a"),
        bgcolor="rgba(255,255,255,0.78)",
    )

    # ---------------------------
    # VETORES Ex, Ey e E em P
    # ---------------------------
    Ex = calc["Ex_dist"]
    Ey = calc["Ey_dist"]
    E = calc["E"]

    vec_ref = max(abs(Ex), abs(Ey), E, 1e-12)
    L = 0.42 * span

    def add_vec(dx, dy, name, color):
        if abs(dx) < 1e-15 and abs(dy) < 1e-15:
            return
        scale = L / vec_ref
        fig.add_annotation(
            x=dx * scale,
            y=a + dy * scale,
            ax=0,
            ay=a,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=3,
            arrowcolor=color,
            text=name,
            font=dict(color=color, size=14),
            bgcolor="rgba(255,255,255,0.68)",
        )

    add_vec(Ex, 0, "Eₓ", "#ff7f0e")
    add_vec(0, Ey, "Eᵧ", "#17a2b8")
    add_vec(Ex, Ey, "E", "#111111")

    # ---------------------------
    # BOX INFORMATIVO 1 (λ, θ1, θ2)
    # ---------------------------
    if lambda_lin == 0:
        lambda_text = "λ = 0 C/m"
    else:
        sinal = "+" if lambda_lin > 0 else "−"
        lambda_text = f"λ = {sinal}{fmt_num(abs(lambda_lin), 4, 'C/m')}"

    box1 = (
        f"<b>{lambda_text}</b><br>"
        f"θ₁ = {calc['theta1_deg']:.2f}°<br>"
        f"θ₂ = {calc['theta2_deg']:.2f}°"
    )

    fig.add_annotation(
        x=left_lim + 0.03 * (right_lim - left_lim),
        y=top_lim - 0.05 * (top_lim - bottom_lim),
        xanchor="left",
        yanchor="top",
        text=box1,
        align="left",
        showarrow=False,
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
        font=dict(size=14),
    )

    # ---------------------------
    # BOX INFORMATIVO 2 (Ex, Ey, E)
    # ---------------------------
    box2 = (
        f"<b>Componentes do campo</b><br>"
        f"Eₓ = {fmt_num(Ex, 4, 'N/C')}<br>"
        f"Eᵧ = {fmt_num(Ey, 4, 'N/C')}<br>"
        f"|E| = {fmt_num(E, 4, 'N/C')}"
    )

    fig.add_annotation(
        x=right_lim - 0.03 * (right_lim - left_lim),
        y=top_lim - 0.05 * (top_lim - bottom_lim),
        xanchor="right",
        yanchor="top",
        text=box2,
        align="left",
        showarrow=False,
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
        font=dict(size=14),
    )

    # ---------------------------
    # AJUSTES FINAIS
    # ---------------------------
    fig.update_xaxes(
        visible=False,
        range=[left_lim, right_lim],
        fixedrange=False,
    )
    fig.update_yaxes(
        visible=False,
        range=[bottom_lim, top_lim],
        scaleanchor="x",
        scaleratio=1,
        fixedrange=False,
    )

    fig.update_layout(
        width=1100,
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",
    )

    return fig


def plotly_scrollable(fig):
    """
    Renderiza a figura em um contêiner com rolagem horizontal,
    útil para telas pequenas/celular.
    """
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displayModeBar": False,
            "scrollZoom": True,
            "responsive": False,
            "doubleClick": "reset",
        },
    )

    wrapped = f"""
    <div style="overflow-x:auto; width:100%; -webkit-overflow-scrolling: touch; border-radius:12px;">
        <div style="min-width:1100px; touch-action: pan-x pan-y;">
            {html}
        </div>
    </div>
    """

    components.html(wrapped, height=650, scrolling=True)


# ============================================================
# CABEÇALHO
# ============================================================
col1, col2 = st.columns([1, 3])

with col1:
    logo = maybe_logo()
    if logo is not None:
        st.image(logo, use_container_width=True)
    else:
        st.info("Adicione o arquivo `logo_maua.png` na raiz do repositório para exibir o logo.")

with col2:
    st.title("Simulador Campo Elétrico Fio")
    st.write("Estude o campo elétrico gerado por um fio em um ponto P a certa distância do fio.")


# ============================================================
# PARÂMETROS
# ============================================================
st.header("Parâmetros")

p1, p2 = st.columns(2)

with p1:
    lambda_lin = st.slider(
        "Densidade linear λ (C/m)",
        min_value=-10e-6,
        max_value=10e-6,
        value=2e-6,
        step=0.1e-6,
        format="%.7f",
    )

    a = st.slider(
        "Distância do fio a (m)",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )

with p2:
    x1 = finite_or_inf_ui("Comprimento x1 (m)", "x1", 4.0)
    x2 = finite_or_inf_ui("Comprimento x2 (m)", "x2", 6.0)

calc = calcular_campo(lambda_lin, a, x1, x2)


# ============================================================
# IMAGEM
# ============================================================
st.header("Imagem")
st.markdown(
    '<div class="scroll-hint">Em celular, deslize horizontalmente sobre a figura para enxergar tudo.</div>',
    unsafe_allow_html=True,
)

fig = build_plot(lambda_lin, a, x1, x2, calc)
plotly_scrollable(fig)


# ============================================================
# EQUAÇÕES
# ============================================================
st.header("Equações")

st.subheader("Ângulos")
st.latex(r"\theta_1 = \arctan\left(\frac{x_1}{a}\right)")
st.latex(r"\theta_2 = \arctan\left(\frac{x_2}{a}\right)")

st.subheader("Componentes do campo elétrico em função das distâncias")
st.latex(
    r"E_x = \frac{\lambda}{4\pi\varepsilon_0}"
    r"\left(\frac{1}{\sqrt{x_2^2+a^2}} - \frac{1}{\sqrt{x_1^2+a^2}}\right)"
)
st.latex(
    r"E_y = \frac{\lambda}{4\pi\varepsilon_0\,a}"
    r"\left(\frac{x_2}{\sqrt{x_2^2+a^2}} + \frac{x_1}{\sqrt{x_1^2+a^2}}\right)"
)

st.subheader("Componentes do campo elétrico em função dos ângulos")
st.latex(
    r"E_x = \frac{\lambda}{4\pi\varepsilon_0\,a}"
    r"\left(\cos\left|\theta_2\right| - \cos\left|\theta_1\right|\right)"
)
st.latex(
    r"E_y = \frac{\lambda}{4\pi\varepsilon_0\,a}"
    r"\left(\sin\left|\theta_1\right| + \sin\left|\theta_2\right|\right)"
)

st.subheader("Módulo do campo elétrico")
st.latex(r"E = \sqrt{E_x^2 + E_y^2}")

st.subheader("Permissividade do vácuo")
st.latex(r"\varepsilon_0 = 8.8541878128\times 10^{-12}\ \text{F/m}")


# ============================================================
# CÁLCULOS
# ============================================================
st.header("Cálculos")

# ---------------------------
# 1) Ângulos
# ---------------------------
st.subheader("1) Ângulos")

if x1 is None:
    st.latex(
        r"\theta_1 = \arctan\left(\frac{x_1}{a}\right),\quad x_1\to\infty \Rightarrow \theta_1 \to 90^\circ"
    )
else:
    st.latex(
        rf"\theta_1 = \arctan\left(\frac{{{fmt_num_plain(x1,4)}}}{{{fmt_num_plain(a,4)}}}\right)"
        rf" = {fmt_num_plain(calc['theta1_deg'],4)}^\circ"
    )

if x2 is None:
    st.latex(
        r"\theta_2 = \arctan\left(\frac{x_2}{a}\right),\quad x_2\to\infty \Rightarrow \theta_2 \to 90^\circ"
    )
else:
    st.latex(
        rf"\theta_2 = \arctan\left(\frac{{{fmt_num_plain(x2,4)}}}{{{fmt_num_plain(a,4)}}}\right)"
        rf" = {fmt_num_plain(calc['theta2_deg'],4)}^\circ"
    )


# ---------------------------
# 2) Componentes em função das distâncias
# ---------------------------
st.subheader("2) Componentes em função das distâncias")

# Ex
if x1 is None or x2 is None:
    linhas_ex = [
        r"E_x = \frac{\lambda}{4\pi\varepsilon_0}\left(\frac{1}{\sqrt{x_2^2+a^2}} - \frac{1}{\sqrt{x_1^2+a^2}}\right)"
    ]
    if x2 is None:
        linhas_ex.append(r"\frac{1}{\sqrt{x_2^2+a^2}} \xrightarrow[x_2\to\infty]{} 0")
    if x1 is None:
        linhas_ex.append(r"\frac{1}{\sqrt{x_1^2+a^2}} \xrightarrow[x_1\to\infty]{} 0")
    st.latex(r"\\ ".join(linhas_ex))

termo_x2_ex = (
    "0"
    if x2 is None
    else rf"\frac{{1}}{{\sqrt{{{fmt_num_plain(x2,4)}^2 + {fmt_num_plain(a,4)}^2}}}}"
)

termo_x1_ex = (
    "0"
    if x1 is None
    else rf"\frac{{1}}{{\sqrt{{{fmt_num_plain(x1,4)}^2 + {fmt_num_plain(a,4)}^2}}}}"
)

st.latex(
    rf"E_x = \frac{{{fmt_num_plain(lambda_lin,6)}}}{{4\pi\,{fmt_num_plain(EPS0,6)}}}"
    rf"\left({termo_x2_ex} - {termo_x1_ex}\right)"
    rf" = {fmt_num_plain(calc['Ex_dist'],6)}\ \text{{N/C}}"
)

# Ey
if x1 is None or x2 is None:
    linhas_ey = [
        r"E_y = \frac{\lambda}{4\pi\varepsilon_0 a}\left(\frac{x_2}{\sqrt{x_2^2+a^2}} + \frac{x_1}{\sqrt{x_1^2+a^2}}\right)"
    ]
    if x2 is None:
        linhas_ey.append(
            r"\frac{x_2}{\sqrt{x_2^2+a^2}} = \frac{1}{\sqrt{1 + a^2/x_2^2}} \xrightarrow[x_2\to\infty]{} 1"
        )
    if x1 is None:
        linhas_ey.append(
            r"\frac{x_1}{\sqrt{x_1^2+a^2}} = \frac{1}{\sqrt{1 + a^2/x_1^2}} \xrightarrow[x_1\to\infty]{} 1"
        )
    st.latex(r"\\ ".join(linhas_ey))

termo_x2_ey = (
    "1"
    if x2 is None
    else rf"\frac{{{fmt_num_plain(x2,4)}}}{{\sqrt{{{fmt_num_plain(x2,4)}^2 + {fmt_num_plain(a,4)}^2}}}}"
)

termo_x1_ey = (
    "1"
    if x1 is None
    else rf"\frac{{{fmt_num_plain(x1,4)}}}{{\sqrt{{{fmt_num_plain(x1,4)}^2 + {fmt_num_plain(a,4)}^2}}}}"
)

st.latex(
    rf"E_y = \frac{{{fmt_num_plain(lambda_lin,6)}}}{{4\pi\,{fmt_num_plain(EPS0,6)}\,{fmt_num_plain(a,4)}}}"
    rf"\left({termo_x2_ey} + {termo_x1_ey}\right)"
    rf" = {fmt_num_plain(calc['Ey_dist'],6)}\ \text{{N/C}}"
)


# ---------------------------
# 3) Componentes em função dos ângulos
# ---------------------------
st.subheader("3) Componentes em função dos ângulos")

st.latex(
    rf"E_x = \frac{{{fmt_num_plain(lambda_lin,6)}}}{{4\pi\,{fmt_num_plain(EPS0,6)}\,{fmt_num_plain(a,4)}}}"
    rf"\left(\cos {fmt_num_plain(calc['theta2_deg'],4)}^\circ - \cos {fmt_num_plain(calc['theta1_deg'],4)}^\circ\right)"
    rf" = {fmt_num_plain(calc['Ex_ang'],6)}\ \text{{N/C}}"
)

st.latex(
    rf"E_y = \frac{{{fmt_num_plain(lambda_lin,6)}}}{{4\pi\,{fmt_num_plain(EPS0,6)}\,{fmt_num_plain(a,4)}}}"
    rf"\left(\sin {fmt_num_plain(calc['theta1_deg'],4)}^\circ + \sin {fmt_num_plain(calc['theta2_deg'],4)}^\circ\right)"
    rf" = {fmt_num_plain(calc['Ey_ang'],6)}\ \text{{N/C}}"
)


# ---------------------------
# 4) Módulo e direção do campo
# ---------------------------
st.subheader("4) Módulo e direção do campo elétrico")

st.latex(
    rf"E = \sqrt{{\left({fmt_num_plain(calc['Ex_dist'],6)}\right)^2 + \left({fmt_num_plain(calc['Ey_dist'],6)}\right)^2}}"
    rf" = {fmt_num_plain(calc['E'],6)}\ \text{{N/C}}"
)

st.latex(
    rf"\varphi = \operatorname{{atan2}}(E_y, E_x)"
    rf" = \operatorname{{atan2}}\left({fmt_num_plain(calc['Ey_dist'],6)}, {fmt_num_plain(calc['Ex_dist'],6)}\right)"
    rf" = {fmt_num_plain(calc['phi_deg'],4)}^\circ"
)

st.caption("Ângulo \\(\\varphi\\) medido a partir do eixo +x, no sentido anti-horário.")


# ============================================================
# OBSERVAÇÕES
# ============================================================
with st.expander("Observações de uso"):
    st.markdown(
        """
- Para **x₁ infinito**, o aplicativo interpreta o fio estendendo-se indefinidamente para a esquerda.
- Para **x₂ infinito**, o aplicativo interpreta o fio estendendo-se indefinidamente para a direita.
- Quando **λ < 0**, o sentido do campo elétrico é invertido, e o fio fica azul.
- Quando **λ = 0**, o campo é nulo, e o fio fica preto.
- A figura é desenhada com largura grande e contêiner com rolagem horizontal para facilitar uso no celular.
        """
    )
