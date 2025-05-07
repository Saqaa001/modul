import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit  # Logistic function

def main():
    st.title("Rasch Model for Dichotomous Data")
    
    # --- Mathematical Formulation ---
    st.markdown("""
    ## Mathematical Formulation
    The Rasch model specifies the probability that a person $n$ answers item $i$ correctly as:
    
    $$
    P(X_{ni} = 1 | \\theta_n, \\beta_i) = \\frac{e^{(\\theta_n - \\beta_i)}}{1 + e^{(\\theta_n - \\beta_i)}}
    $$
    
    Where:
    - $X_{ni}$ is the response (1 = correct, 0 = incorrect)
    - $\\theta_n$ = ability of person $n$
    - $\\beta_i$ = difficulty of item $i$
    """)
    
    st.markdown("---")

    # --- Example Student Data ---
    st.header("📊 Example Student Data")
    
    # Define example students
    students = [
        {"Student": "Ali", "θ": -1.2, "Responses": [0, 1, 0]},
        {"Student": "Vali", "θ": 0.5, "Responses": [1, 1, 0]},
        {"Student": "Sali", "θ": 2.0, "Responses": [1, 1, 1]}
    ]
    
    # Define example items
    items = [
        {"Item": "Q1 (Easy)", "β": -1.0},
        {"Item": "Q2 (Medium)", "β": 0.0},
        {"Item": "Q3 (Hard)", "β": 1.5}
    ]
    
    # Display student and item tables side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Students")
        st.table(pd.DataFrame(students)[["Student", "θ"]])
    
    with col2:
        st.subheader("Items")
        st.table(pd.DataFrame(items))
    
    # --- Interactive Visualization ---
    st.header("📈 Interactive ICC Curve")
    
    # Dropdown to select a student
    selected_student = st.selectbox(
        "Select a student:",
        options=[s["Student"] for s in students],
        index=0
    )
    
    # Get θ for selected student
    student_θ = next(s["θ"] for s in students if s["Student"] == selected_student)
    
    # Plot ICC for all items
    θ_range = np.linspace(-3, 3, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for item in items:
        β = item["β"]
        prob = expit(θ_range - β)
        ax.plot(θ_range, prob, label=f"{item['Item']} (β={β})")
    
    # Mark student's ability level
    ax.axvline(x=student_θ, color='red', linestyle='--', label=f"{selected_student}'s θ = {student_θ}")
    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("P(X=1)")
    ax.set_title("Item Characteristic Curves (ICC)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    
    # --- Probability Table for Selected Student ---
    st.subheader(f"🔍 Predicted Probabilities for {selected_student} (θ = {student_θ})")
    
    # Calculate probabilities for each item
    prob_data = []
    for item in items:
        β = item["β"]
        p = expit(student_θ - β)
        prob_data.append({
            "Item": item["Item"],
            "β": β,
            "P(X=1)": f"{p:.2f}",
            "Expected Response": "1" if p >= 0.5 else "0"
        })
    
    st.table(pd.DataFrame(prob_data))

if __name__ == "__main__":
    main()