# CGM Insights Dashboard

A Streamlit web app that transforms raw Continuous Glucose Monitor (CGM) data into actionable insights. Upload your CGM CSV file to visualize time-in-range percentages, glucose trends, spike patterns, and weekly behavior. Built for people with diabetes and healthcare analysts who want more than what Clarity or LibreView provide.

## Features
- 📈 Time-series glucose visualization
- 🎯 Customizable target range sliders
- 📅 Day-by-day and weekday-based analysis
- 🔄 Spike/dip frequency and magnitude detection
- 🧠 Estimated meal times based on glucose acceleration
- 📊 Time in range (TIR), GMI, SD, and more

## Getting Started
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
