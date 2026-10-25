
import os
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from Task import (
    read_files,
    add_signals,
    sub_signals,
    mul_const,
    squaring,
    normalization,
    accum,
    calc_quant,
    encoded_sig,
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Signal Processing Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. THEME
# =========================================================

with st.sidebar:
    st.title("⚙️ Settings")
    theme = st.radio(
        "Appearance",
        ["Light Mode", "Dark Mode"],
        horizontal=True,
    )

if theme == "Light Mode":
    COLORS = {
        "background": "#F5F9FF",
        "card": "#FFFFFF",
        "primary": "#3981D5",
        "heading": "#245CA6",
        "section": "#E7F2FF",
        "teal": "#16A6A1",
        "text": "#26374A",
        "secondary": "#64748B",
        "border": "#D8E5F3",
        "grid": "#D8E5F3",
    }
    mpl_style = "default"
else:
    COLORS = {
        "background": "#0F172A",
        "card": "#182338",
        "primary": "#60A5FA",
        "heading": "#93C5FD",
        "section": "#1E293B",
        "teal": "#2DD4BF",
        "text": "#E5EDF8",
        "secondary": "#A7B4C8",
        "border": "#334155",
        "grid": "#334155",
    }
    mpl_style = "dark_background"


st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
    }}

    h1, h2, h3 {{
        color: {COLORS['heading']};
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS['card']};
    }}

    [data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        padding: 16px;
        border-radius: 12px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {COLORS['border']};
        border-radius: 12px;
    }}

    div.stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 40px;
    }}

    div.stButton > button:hover {{
        border: 1px solid {COLORS['teal']};
        color: white;
        filter: brightness(1.08);
    }}

    div[data-baseweb="input"],
    div[data-baseweb="select"] {{
        border-radius: 8px;
    }}

    hr {{
        border-color: {COLORS['border']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD SIGNALS
# =========================================================

INPUT_FOLDER = "input"

try:
    signals, quant_signals = read_files(INPUT_FOLDER)
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Could not load the input signals: {exc}")
    st.info(
        "Expected structure: input/operation_input and "
        "input/quant_input, each containing .txt files."
    )
    st.stop()

if not signals and not quant_signals:
    st.error("No signals were loaded. Check your input folders.")
    st.stop()

# Quantization uses quant_input signals when available.
# Otherwise, it falls back to the operation signals.
quant_source = quant_signals if quant_signals else signals


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def compatible_signals(signal_dict, names):
    """Check that selected signals have matching indices and lengths."""

    if len(names) < 2:
        return False, "Please select at least two signals."

    reference_indices = signal_dict[names[0]]["indices"]
    reference_samples = signal_dict[names[0]]["samples"]

    if not reference_samples:
        return False, "The selected signals contain no samples."

    for name in names:
        indices = signal_dict[name]["indices"]
        samples = signal_dict[name]["samples"]

        if len(indices) != len(samples):
            return False, f"{name} has inconsistent indices and samples."

        if indices != reference_indices:
            return False, (
                "All selected signals must have identical sample indices."
            )

        if len(samples) != len(reference_samples):
            return False, "The selected signals have different lengths."

    return True, ""


def plot_signal(indices, samples, title, discrete=False):
    """Display a consistently sized signal plot."""

    with plt.style.context(mpl_style):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))

        if discrete:
            markerline, stemlines, baseline = ax.stem(
                indices,
                samples,
                linefmt="-",
                markerfmt="o",
                basefmt="-",
            )
            plt.setp(markerline, markersize=5)
            plt.setp(stemlines, linewidth=1.5)
            plt.setp(baseline, linewidth=0.8)
        else:
            ax.plot(
                indices,
                samples,
                linewidth=2,
                marker="o",
                markersize=3,
            )

        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)

        ax.set_facecolor(COLORS["card"])
        fig.patch.set_facecolor(COLORS["card"])

        ax.tick_params(colors=COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["heading"])

        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def show_result(indices, samples, title, discrete=True):
    """Show a result with its plot and sample values."""

    if samples is None:
        st.warning("No result is available.")
        return

    if len(indices) != len(samples):
        st.error("The result indices and sample values do not match.")
        return

    if len(samples) == 0:
        st.warning("The result contains no samples.")
        return

    st.markdown(f"#### {title}")
    plot_signal(indices, samples, title, discrete=discrete)

    with st.expander("View numerical samples"):
        st.dataframe(
            {
                "Index": indices,
                "Sample Value": samples,
            },
            use_container_width=True,
            hide_index=True,
        )


def remember_result(key, result):
    """Keep an operation result across Streamlit reruns."""
    st.session_state[key] = result


def show_saved_result(key):
    result = st.session_state.get(key)

    if result is not None:
        st.divider()
        show_result(
            result["indices"],
            result["samples"],
            result["title"],
        )


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.markdown(
    f"""
    <div style="
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
    ">
        <h1 style="margin: 0;">📈 Signal Processing Framework</h1>
        <p style="color: {COLORS['secondary']}; margin-bottom: 0;">
            Explore, visualize, and process discrete-time signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 6. SIGNAL SELECTION
# =========================================================

st.header("1. Signal Selection")

if signals:
    signal_names = list(signals.keys())

    selected_signal = st.selectbox(
        "Select Input Signal",
        options=signal_names,
        key="main_signal",
    )

    # -----------------------------------------------------
    # TASK 1: SIGNAL DISPLAY
    # -----------------------------------------------------

    st.header("2. Signal Display")

    display_col1, display_col2 = st.columns(2)

    with display_col1:
        with st.container(border=True):
            st.markdown("#### Continuous Signal")
            plot_signal(
                signals[selected_signal]["indices"],
                signals[selected_signal]["samples"],
                f"{selected_signal} — Continuous",
                discrete=False,
            )

    with display_col2:
        with st.container(border=True):
            st.markdown("#### Discrete Signal")
            plot_signal(
                signals[selected_signal]["indices"],
                signals[selected_signal]["samples"],
                f"{selected_signal} — Discrete",
                discrete=True,
            )

else:
    selected_signal = None
    signal_names = []
    st.warning("No operation signals are available.")


# =========================================================
# 7. ARITHMETIC OPERATIONS
# =========================================================

st.divider()
st.header("3. Arithmetic Operations")

if signals:

    (
        addition_tab,
        subtraction_tab,
        multiplication_tab,
        squaring_tab,
        normalization_tab,
        accumulation_tab,
    ) = st.tabs(
        [
            "Addition",
            "Subtraction",
            "Multiplication",
            "Squaring",
            "Normalization",
            "Accumulation",
        ]
    )

    # -----------------------------------------------------
    # TAB 1: ADDITION
    # -----------------------------------------------------

    with addition_tab:
        st.subheader("Addition")

        addition_names = st.multiselect(
            "Select two or more signals",
            options=signal_names,
            default=[selected_signal] if selected_signal else [],
            key="addition_names",
        )

        if st.button("Run Addition", key="run_addition"):
            valid, message = compatible_signals(signals, addition_names)

            if not valid:
                st.warning(message)
            else:
                try:
                    result_indices, result_samples = add_signals(
                        signals,
                        addition_names,
                    )

                    remember_result(
                        "addition_result",
                        {
                            "indices": result_indices,
                            "samples": result_samples,
                            "title": "Addition Result",
                        },
                    )
                    st.success("Addition completed successfully.")

                except (ValueError, IndexError, KeyError, TypeError) as exc:
                    st.error(f"Addition failed: {exc}")

        show_saved_result("addition_result")

    # -----------------------------------------------------
    # TAB 2: SUBTRACTION
    # -----------------------------------------------------

    with subtraction_tab:
        st.subheader("Subtraction")

        subtraction_names = st.multiselect(
            "Select signals to subtract",
            options=signal_names,
            default=[selected_signal] if selected_signal else [],
            key="subtraction_names",
            help=(
                "The existing subtraction function uses absolute "
                "differences between the selected signals."
            ),
        )

        if st.button("Run Subtraction", key="run_subtraction"):
            valid, message = compatible_signals(
                signals,
                subtraction_names,
            )

            if not valid:
                st.warning(message)
            else:
                try:
                    result_indices, result_samples = sub_signals(
                        signals,
                        subtraction_names,
                    )

                    remember_result(
                        "subtraction_result",
                        {
                            "indices": result_indices,
                            "samples": result_samples,
                            "title": "Subtraction Result",
                        },
                    )
                    st.success("Subtraction completed successfully.")

                except (ValueError, IndexError, KeyError, TypeError) as exc:
                    st.error(f"Subtraction failed: {exc}")

        show_saved_result("subtraction_result")

    # -----------------------------------------------------
    # TAB 3: MULTIPLICATION
    # -----------------------------------------------------

    with multiplication_tab:
        st.subheader("Multiplication by a Constant")

        multiplication_signal = st.selectbox(
            "Select Signal",
            options=signal_names,
            index=(
                signal_names.index(selected_signal)
                if selected_signal in signal_names
                else 0
            ),
            key="multiplication_signal",
        )

        constant = st.number_input(
            "Enter Constant",
            value=1.0,
            step=0.5,
            key="multiplication_constant",
        )

        if st.button("Run Multiplication", key="run_multiplication"):
            try:
                result_indices, result_samples = mul_const(
                    constant,
                    signals,
                    multiplication_signal,
                )

                remember_result(
                    "multiplication_result",
                    {
                        "indices": result_indices,
                        "samples": result_samples,
                        "title": f"Multiplication Result (× {constant:g})",
                    },
                )
                st.success("Multiplication completed successfully.")

            except (ValueError, IndexError, KeyError, TypeError) as exc:
                st.error(f"Multiplication failed: {exc}")

        show_saved_result("multiplication_result")

    # -----------------------------------------------------
    # TAB 4: SQUARING
    # -----------------------------------------------------

    with squaring_tab:
        st.subheader("Squaring")

        square_signal = st.selectbox(
            "Select Signal",
            options=signal_names,
            index=(
                signal_names.index(selected_signal)
                if selected_signal in signal_names
                else 0
            ),
            key="squaring_signal",
        )

        if st.button("Run Squaring", key="run_squaring"):
            try:
                result_indices, result_samples = squaring(
                    signals,
                    square_signal,
                )

                remember_result(
                    "squaring_result",
                    {
                        "indices": result_indices,
                        "samples": result_samples,
                        "title": "Squaring Result",
                    },
                )
                st.success("Squaring completed successfully.")

            except (ValueError, IndexError, KeyError, TypeError) as exc:
                st.error(f"Squaring failed: {exc}")

        show_saved_result("squaring_result")

    # -----------------------------------------------------
    # TAB 5: NORMALIZATION
    # -----------------------------------------------------

    with normalization_tab:
        st.subheader("Normalization")

        normalization_signal = st.selectbox(
            "Select Signal",
            options=signal_names,
            index=(
                signal_names.index(selected_signal)
                if selected_signal in signal_names
                else 0
            ),
            key="normalization_signal",
        )

        normalization_method = st.radio(
            "Target Range",
            options=["[-1, 1]", "[0, 1]"],
            horizontal=True,
            key="normalization_method",
        )

        if normalization_method == "[-1, 1]":
            normalization_type = "minus_one_to_one"
        else:
            normalization_type = "zero_to_one"

        if st.button("Run Normalization", key="run_normalization"):
            try:
                result = normalization(
                    normalization_type,
                    signals,
                    normalization_signal,
                )

                if result is None:
                    st.warning(
                        "Normalization is not possible for a "
                        "constant-valued signal."
                    )
                else:
                    result_indices, result_samples = result

                    remember_result(
                        "normalization_result",
                        {
                            "indices": result_indices,
                            "samples": result_samples,
                            "title": (
                                f"Normalized Signal — "
                                f"{normalization_method}"
                            ),
                        },
                    )
                    st.success("Normalization completed successfully.")

            except (ValueError, IndexError, KeyError, TypeError) as exc:
                st.error(f"Normalization failed: {exc}")

        show_saved_result("normalization_result")

    # -----------------------------------------------------
    # TAB 6: ACCUMULATION
    # -----------------------------------------------------

    with accumulation_tab:
        st.subheader("Accumulation")

        accumulation_signal = st.selectbox(
            "Select Signal",
            options=signal_names,
            index=(
                signal_names.index(selected_signal)
                if selected_signal in signal_names
                else 0
            ),
            key="accumulation_signal",
        )

        if st.button("Run Accumulation", key="run_accumulation"):
            try:
                result_indices, result_samples = accum(
                    signals,
                    accumulation_signal,
                )

                remember_result(
                    "accumulation_result",
                    {
                        "indices": result_indices,
                        "samples": result_samples,
                        "title": "Accumulation Result",
                    },
                )
                st.success("Accumulation completed successfully.")

            except (ValueError, IndexError, KeyError, TypeError) as exc:
                st.error(f"Accumulation failed: {exc}")

        show_saved_result("accumulation_result")


# =========================================================
# 8. QUANTIZATION
# =========================================================

st.divider()
st.header("4. Quantization")

st.markdown(
    "Choose an input signal and a quantization method, "
    "then inspect the quantized values, signed errors, "
    "binary encoding, and average error power."
)

if not quant_source:
    st.warning(
        "No quantization signals were found. "
        "Add .txt files to input/quant_input or use the operation signals."
    )

else:
    quant_signal_names = list(quant_source.keys())

    q_col1, q_col2 = st.columns(2)

    with q_col1:
        quant_signal = st.selectbox(
            "Select Input Signal",
            options=quant_signal_names,
            key="quant_signal",
        )

    with q_col2:
        quant_method = st.radio(
            "Quantization Method",
            options=["Number of Bits", "Number of Levels"],
            horizontal=True,
            key="quant_method",
        )

    # -----------------------------------------------------
    # PARAMETER INPUT
    # -----------------------------------------------------

    if quant_method == "Number of Bits":
        bits = st.number_input(
            "Enter Number of Bits",
            min_value=1,
            max_value=16,
            value=3,
            step=1,
            key="quant_bits",
        )

        num_type = "bits"
        quant_parameter = int(bits)
        num_levels = 2 ** int(bits)

        st.info(
            f"Number of quantization levels: "
            f"**L = 2^{int(bits)} = {num_levels}**"
        )

    else:
        levels = st.number_input(
            "Enter Number of Levels",
            min_value=2,
            max_value=65536,
            value=8,
            step=1,
            key="quant_levels",
        )

        num_type = "levels"
        quant_parameter = int(levels)
        num_levels = int(levels)
        encoding_width = math.ceil(math.log2(num_levels))

        st.info(
            f"Number of levels: **{num_levels}** | "
            f"Binary encoding width: **{encoding_width} bits**"
        )

        if num_levels & (num_levels - 1):
            st.caption(
                "Since the number of levels is not a power of two, "
                "some binary codes will remain unused."
            )

    # -----------------------------------------------------
    # RUN QUANTIZATION
    # -----------------------------------------------------

    if st.button(
        "Run Quantization",
        key="run_quantization",
        use_container_width=True,
    ):
        samples = quant_source[quant_signal]["samples"]
        indices = quant_source[quant_signal]["indices"]

        if not samples:
            st.error("The selected signal has no samples.")

        elif len(indices) != len(samples):
            st.error("The signal indices and samples have different lengths.")

        elif not np.all(np.isfinite(np.asarray(samples, dtype=float))):
            st.error("The signal contains non-finite sample values.")

        else:
            try:
                quantized_samples, ranges, level_indices = calc_quant(
                    num_type,
                    quant_parameter,
                    quant_source,
                    quant_signal,
                )

                # Validate that every sample was quantized.
                if (
                    len(quantized_samples) != len(samples)
                    or len(level_indices) != len(samples)
                ):
                    raise ValueError(
                        "Some samples could not be assigned "
                        "to quantization levels."
                    )

                # Signed quantization error:
                # e[n] = x[n] - x_q[n]
                errors = [
                    original - quantized
                    for original, quantized
                    in zip(samples, quantized_samples)
                ]

                # Mean squared quantization error:
                # Pe = (1/N) * sum(e[n]**2)
                error_power = float(np.mean(np.square(errors)))

                # Reuse the existing encoder for the binary codes.
                encoded_samples = encoded_sig(
                    num_type,
                    quant_parameter,
                    quant_source,
                    quant_signal,
                )

                if len(encoded_samples) != len(samples):
                    raise ValueError(
                        "The encoded sequence length does not match "
                        "the input signal."
                    )

                remember_result(
                    "quantization_result",
                    {
                        "signal_name": quant_signal,
                        "method": num_type,
                        "parameter": quant_parameter,
                        "indices": list(indices),
                        "original": list(samples),
                        "quantized": list(quantized_samples),
                        "errors": errors,
                        "levels": list(level_indices),
                        "encoded": list(encoded_samples),
                        "error_power": error_power,
                        "ranges": ranges,
                    },
                )

                st.success("Quantization completed successfully.")

            except (
                ValueError,
                IndexError,
                KeyError,
                TypeError,
                ZeroDivisionError,
                OverflowError,
            ) as exc:
                st.error(f"Quantization failed: {exc}")

    # -----------------------------------------------------
    # DISPLAY QUANTIZATION RESULTS
    # -----------------------------------------------------

    q_result = st.session_state.get("quantization_result")

    # Show results only when they correspond to the current inputs.
    if q_result is not None:
        same_configuration = (
            q_result["signal_name"] == quant_signal
            and q_result["method"] == num_type
            and q_result["parameter"] == quant_parameter
        )

        if same_configuration:
            st.divider()
            st.subheader("Quantization Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                show_result(
                    q_result["indices"],
                    q_result["quantized"],
                    "Quantized Signal",
                    discrete=True,
                )

            with result_col2:
                show_result(
                    q_result["indices"],
                    q_result["errors"],
                    "Quantization Error Signal",
                    discrete=True,
                )

            # ---------------------------------------------
            # BINARY ENCODING TABLE
            # ---------------------------------------------

            st.subheader("Encoded Binary Signal")

            encoding_data = {
                "Sample Index": q_result["indices"],
                "Original Value": q_result["original"],
                "Quantization Level": q_result["levels"],
                "Quantized Value": q_result["quantized"],
                "Binary Code": q_result["encoded"],
            }

            with st.container(border=True):
                st.dataframe(
                    encoding_data,
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

                csv_data = __import__("pandas").DataFrame(
                    encoding_data
                ).to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Encoding Table (CSV)",
                    data=csv_data,
                    file_name="quantization_results.csv",
                    mime="text/csv",
                    key="download_quantization",
                )

            # ---------------------------------------------
            # QUANTIZATION LEVELS
            # ---------------------------------------------

            with st.expander("View Quantization Level Ranges"):
                st.dataframe(
                    q_result["ranges"],
                    use_container_width=True,
                    hide_index=True,
                )

            # ---------------------------------------------
            # AVERAGE ERROR POWER
            # ---------------------------------------------

            st.subheader("Average Power of Error")

            metric_col1, metric_col2 = st.columns([1, 2])

            with metric_col1:
                st.metric(
                    label="Mean Squared Quantization Error",
                    value=f"{q_result['error_power']:.8f}",
                )

            with metric_col2:
                st.markdown(
                    r"""
                    \[
                    P_e = \frac{1}{N}
                    \sum_{n=0}^{N-1} e[n]^2
                    \]
                    """
                )

                st.caption(
                    "Calculated from the signed error samples. "
                    "The displayed metric is the mean of their squares."
                )

        else:
            st.info(
                "The quantization inputs have changed. "
                "Click **Run Quantization** to calculate updated results."
            )


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Digital Signal Processing Framework | "
    "Task 1: Signal Operations | Task 2: Quantization"
)
