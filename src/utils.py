import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(actual, predicted, ticker="AAPL"):
    """Build the figure and render it in Streamlit if available, else show it."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(predicted, label='Predicted', color='red')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    fig.tight_layout()

    try:
        import streamlit as st
        st.pyplot(fig)
    except Exception:
        fig.savefig("prediction.png")
    finally:
        plt.close(fig)
