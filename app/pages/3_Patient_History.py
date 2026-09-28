"""
Patient History page -- a searchable list of every real patient logged in
the local database, with their visit count and a shortcut into their
Mode 2 trajectory view.

This is the page that makes the tool feel like something a clinician keeps
open across a workday rather than a one-shot calculator -- see
Product_Story_Build_to_Use.md's Persona 1 section for the full framing.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.storage.db import get_connection

st.set_page_config(page_title="Patient History", page_icon="\U0001F4CB")
st.title("Patient History")
st.caption("Every patient checked in through this app, and how many visits are on record.")

conn = get_connection()
rows = conn.execute(
    """SELECT p.patient_id, p.created_at, COUNT(v.visit_id) as n_visits,
              MAX(v.visit_date) as last_visit
       FROM patients p LEFT JOIN visits v ON p.patient_id = v.patient_id
       GROUP BY p.patient_id ORDER BY last_visit DESC"""
).fetchall()
conn.close()

if not rows:
    st.info("No patients logged yet. Use Mode 1 to check in a patient with a Patient ID.")
else:
    search = st.text_input("Search by Patient ID")
    for patient_id, created_at, n_visits, last_visit in rows:
        if search and search.lower() not in patient_id.lower():
            continue
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.write(f"**{patient_id}**")
            col2.write(f"{n_visits} visit(s)")
            col3.write(f"Last: {last_visit[:10] if last_visit else 'N/A'}")
            if n_visits >= 2:
                st.caption("2+ visits logged -- available in Mode 2's real-patient list.")
