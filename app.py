"""
WHALER – Streamlit analytics dashboard
=====================================

This module implements a complete Streamlit application for the WHALER MVP.  The
goal of the app is to allow online performers to upload a CSV export of their
earnings data and instantly visualise who their top spending customers are and
how their revenue is distributed across different earning types.  The free
version provides basic insights and intentionally withholds some deeper
analytics to encourage upgrades to paid tiers.

Key features of this MVP include:

* Robust CSV ingestion with automatic column normalisation, date parsing,
  numeric conversion and deduplication based on a user‑defined rule
  (Date+User+Amount+Type).  Errors in parsing are handled gracefully and the
  user is informed when data could not be read.
* A high‑level overview card showing total earnings and the number of unique
  whales (customers) in the dataset.
* A ranked leaderboard of the top 10 whales by total spend.  The top three
  whales are highlighted, while ranks 4–10 are obscured to create curiosity
  around the premium tiers.
* A stacked bar chart breaking down spending for the top three whales across
  chat, video, gifts and other earning categories.  Custom colours are used
  to align with the WHALER brand palette.
* Premium feature teaser cards for WHALER Plus and WHALER Pro, which outline
  locked features and encourage users to upgrade.

The application is designed with a dark, modern dashboard aesthetic.  See
`.streamlit/config.toml` for the theme configuration.  See the README in the
associated repository for deployment details.
"""

from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to a canonical form.

    The export files users upload may use different names for the same
    underlying information.  This function attempts to map a variety of
    potential column names to standard names: `Date`, `User`, `Type` and
    `Amount`.  Names are matched case–insensitively and whitespace is
    ignored.  Extra columns are preserved but not used directly.

    Parameters
    ----------
    df:
        The DataFrame whose columns need normalisation.

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with renamed columns.
    """
    mapping = {
        "date": "Date",
        "timestamp": "Date",
        "created_at": "Date",
        "datetime": "Date",
        "user": "User",
        "username": "User",
        "customer": "User",
        "client": "User",
        "payer": "User",
        "type": "Type",
        "category": "Type",
        "payment_type": "Type",
        "amount": "Amount",
        "value": "Amount",
        "price": "Amount",
        "paid": "Amount",
    }
    new_cols = {}
    for col in df.columns:
        clean = re.sub(r"\s+", "", col.strip().lower())
        if clean in mapping:
            new_cols[col] = mapping[clean]
    return df.rename(columns=new_cols)


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the uploaded earnings data.

    This function performs several operations:
    * Normalise column names.
    * Ensure required columns are present; drop rows missing required values.
    * Convert dates to datetime (invalid strings coerced to NaT).
    * Convert amounts to numeric values after stripping currency symbols and
      other non‑numeric characters; invalid entries are coerced to NaN.
    * Coerce unknown earning types into an "Other" category.
    * Deduplicate rows based on Date, User, Amount and Type.

    Parameters
    ----------
    df:
        Raw DataFrame read from a CSV file.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame ready for analysis.  If required columns are
        missing, an empty DataFrame is returned.
    """
    df = normalise_columns(df)

    required_cols = {"Date", "User", "Amount"}
    if not required_cols.issubset(set(df.columns)):
        # Missing essential data; return empty
        return pd.DataFrame(columns=["Date", "User", "Type", "Amount"])

    # Trim whitespace from string columns
    for c in ["User", "Type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Parse dates – invalid strings become NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Remove rows with missing dates
    df = df[df["Date"].notna()]

    # Clean and convert amount
    # Remove anything that is not a digit, decimal point or minus sign
    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df[df["Amount"].notna()]

    # Handle earning types; normalise to known categories
    if "Type" in df.columns:
        type_map = {
            "chat": "Chat",
            "video": "Video",
            "gift": "Gifts",
            "tips": "Gifts",
            "tip": "Gifts",
        }
        def map_type(val: str) -> str:
            lower = str(val).strip().lower()
            for key, out in type_map.items():
                if key in lower:
                    return out
            return "Other"

        df["Type"] = df["Type"].apply(map_type)
    else:
        df["Type"] = "Other"

    # Deduplicate rows based on Date, User, Amount and Type
    df = df.drop_duplicates(subset=["Date", "User", "Amount", "Type"])

    return df


def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the top spenders (whales) in the dataset.

    Groups by user and calculates total spend.  The resulting DataFrame is
    sorted in descending order of spending.

    Parameters
    ----------
    df:
        Cleaned DataFrame with at least 'User' and 'Amount' columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['User', 'Total'] sorted by descending
        spending.
    """
    whales = df.groupby("User", as_index=False)["Amount"].sum()
    whales = whales.rename(columns={"Amount": "Total"})
    whales = whales.sort_values("Total", ascending=False).reset_index(drop=True)
    return whales


def plot_top_whales_breakdown(top3_df: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """Create a stacked bar chart showing spending breakdown for the top three whales.

    Parameters
    ----------
    top3_df:
        DataFrame of the top three whales containing 'User' and 'Total'.
    df:
        Cleaned DataFrame containing all transactions.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure with a stacked bar chart.
    """
    categories = ["Chat", "Video", "Gifts", "Other"]
    colours = {
        "Chat": "#2F80ED",    # Primary blue
        "Video": "#56A0FF",   # Secondary blue
        "Gifts": "#27AE60",   # Green
        "Other": "#2DDAE3",   # Aqua
    }
    # Prepare data matrix for stacking
    data = []
    labels = []
    for _, row in top3_df.iterrows():
        user = row["User"]
        labels.append(user)
        user_df = df[df["User"] == user]
        totals = []
        for cat in categories:
            val = user_df[user_df["Type"] == cat]["Amount"].sum()
            totals.append(val)
        data.append(totals)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(7, 4))
    # Set dark background to harmonise with theme
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    # Plot bars
    bottoms = np.zeros(len(top3_df))
    for i, cat in enumerate(categories):
        ax.bar(labels, data[:, i], bottom=bottoms, color=colours[cat], label=cat)
        bottoms += data[:, i]
    # Styling
    ax.set_title("Top 3 Whales Revenue Breakdown", color="white", fontsize=14, pad=15)
    ax.set_ylabel("Earnings", color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # Rotate labels if needed
    plt.xticks(rotation=0)
    # Legend
    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color("white")
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def render_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Render the top 10 whales leaderboard with blurred lower ranks.

    Parameters
    ----------
    leaderboard:
        DataFrame containing whales sorted by total spend.  Should include at
        least 10 rows; if fewer are provided, all rows are shown without
        blurring.
    """
    max_rows = min(10, len(leaderboard))
    top10 = leaderboard.iloc[:max_rows].copy()
    # Format currency
    top10["Display"] = top10["Total"].apply(lambda x: f"${x:,.0f}")
    # Build HTML table manually for control over styling and blurring
    rows_html = []
    for idx, (_, row) in enumerate(top10.iterrows(), start=1):
        if idx <= 3:
            name_html = f"<span style='font-weight:bold'>{row['User']}</span>"
            total_html = f"<span style='font-weight:bold'>{row['Display']}</span>"
        else:
            # Blur names and totals for curiosity
            name_html = "<span style='filter: blur(4px);'>█████</span>"
            total_html = "<span style='filter: blur(4px);'>█████</span>"
        rows_html.append(
            f"<tr><td style='padding: 4px 8px;'>{idx}</td>"
            f"<td style='padding: 4px 8px;'>{name_html}</td>"
            f"<td style='padding: 4px 8px; text-align:right;'>{total_html}</td></tr>"
        )
    table_html = (
        "<table style='width:100%; border-collapse: collapse; font-size: 0.9rem;'>"
        "<thead><tr>"
        "<th style='text-align:left;'>Rank</th>"
        "<th style='text-align:left;'>Whale</th>"
        "<th style='text-align:right;'>Total</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_premium_teasers() -> None:
    """Display locked preview cards for WHALER Plus and WHALER Pro.

    Each card lists the features unlocked at that tier and uses a muted
    appearance with a lock icon to suggest restricted access.  The design
    encourages users to upgrade without implementing actual billing logic.
    """
    plus_features = [
        "Full top 10 whales", "Daily earnings average", "Weekly projection",
        "Monthly projection", "Yearly projection", "Whale concentration %"
    ]
    pro_features = [
        "Whale retention tracking", "Revenue mix diagnostics",
        "Leach’s list (time wasters)", "Advanced projections",
        "Performer recommendations", "Deeper analytics"
    ]
    # Use two columns to layout the teaser cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div style='border: 1px solid #444; border-radius: 8px; padding: 16px;\n"
            "background-color: #1e2535; height: 100%;'>\n"
            "<h4 style='margin-top:0;'>WHALER Plus</h4>\n"
            "<p style='color:#aaa;'>Unlock these features:</p>\n"
            + "<ul style='padding-left:16px; color:#888;'>" + "".join(
                [f"<li>🔒 {feat}</li>" for feat in plus_features]
            ) + "</ul>\n"
            "<div style='text-align:center; margin-top:8px;'>"
            "<span style='background-color:#2F80ED; color:white; padding:8px 12px;\n"
            "border-radius:4px; cursor:pointer; opacity:0.6;'>Upgrade to Plus</span>"
            "</div>\n"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div style='border: 1px solid #444; border-radius: 8px; padding: 16px;\n"
            "background-color: #1e2535; height: 100%;'>\n"
            "<h4 style='margin-top:0;'>WHALER Pro</h4>\n"
            "<p style='color:#aaa;'>Unlock these features:</p>\n"
            + "<ul style='padding-left:16px; color:#888;'>" + "".join(
                [f"<li>🔒 {feat}</li>" for feat in pro_features]
            ) + "</ul>\n"
            "<div style='text-align:center; margin-top:8px;'>"
            "<span style='background-color:#27AE60; color:white; padding:8px 12px;\n"
            "border-radius:4px; cursor:pointer; opacity:0.6;'>Upgrade to Pro</span>"
            "</div>\n"
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Streamlit WHALER application."""
    st.set_page_config(
        page_title="WHALER – Find your whales", page_icon="🐋", layout="centered"
    )
    # Header
    st.markdown(
        "<h1 style='color:#ffffff; margin-bottom:4px;'>WHALER</h1>\n"
        "<p style='color:#888; margin-top:0;'>Upload your earnings report.\n"
        "Instantly see who is really paying you.</p>",
        unsafe_allow_html=True,
    )
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV earnings export", type=["csv"], help="Only .csv files are supported"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file into a DataFrame
            # Use utf-8-sig to automatically strip BOM if present
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", engine="python")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        df_clean = clean_and_prepare(df_raw)
        if df_clean.empty:
            st.warning(
                "We couldn't find the required columns (Date, User, Amount). "
                "Please ensure your file contains these fields."
            )
            return

        # Compute metrics
        total_earnings = df_clean["Amount"].sum()
        total_whales = df_clean["User"].nunique()
        whales_df = compute_leaderboard(df_clean)

        # Layout: metrics row
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(
            "Total Earnings", f"${total_earnings:,.2f}", help="Sum of all transactions"
        )
        m_col2.metric(
            "Total Whales", total_whales, help="Number of unique spenders"
        )

        # Leaderboard and chart side by side
        lb_col, chart_col = st.columns((1, 1))
        with lb_col:
            st.subheader("Top Whales")
            render_leaderboard(whales_df)
        with chart_col:
            st.subheader("Whale Breakdown")
            if len(whales_df) >= 1:
                top3 = whales_df.head(3)
                fig = plot_top_whales_breakdown(top3, df_clean)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Not enough data for a breakdown chart.")

        # Premium teasers
        st.markdown("---")
        st.subheader("Unlock more insights")
        render_premium_teasers()
    else:
        # Placeholder message when no file is uploaded
        st.info(
            "Upload a CSV export from your platform to discover your whales and revenue patterns."
        )


if __name__ == "__main__":
