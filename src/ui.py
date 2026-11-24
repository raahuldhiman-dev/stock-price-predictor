import streamlit as st

_HIDE = """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {display: none !important;}
</style>
"""


def hide_chrome():
    """Hide Streamlit's menu, deploy button, toolbar and footer."""
    st.markdown(_HIDE, unsafe_allow_html=True)
