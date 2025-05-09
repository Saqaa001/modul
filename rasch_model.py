import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.special import expit # type: ignore
from scipy.stats import norm # type: ignore
import warnings

# aaaSuppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rasch Model Visualizer")

# Sample data from the example
data = {
    "Student": ["Ali", "Vali", "Salim", "Diyor"],
    "Q1: 5 × 6 = ?": [1, 1, 0, 1],
    "Q2: 12 ÷ 3 = ?": [1, 1, 0, 1],
    "Q3: (3 + 2)^2 = ?": [1, 0, 0, 1],
    "Q4: √81 = ?": [1, 1, 0, 1],
    "Q5: 7 + 4 × 2 = ?": [0, 0, 0, 1]
}

def calculate_rasch_parameters(df):
    # Calculate item difficulties
    item_difficulties = {}
    for item in df.columns[1:]:
        p_i = np.clip(df[item].mean(), 0.0001, 0.9999)  # Clip to avoid 0 or 1
        delta_i = np.log((1 - p_i) / p_i)
        item_difficulties[item] = delta_i
    
    # Calculate person abilities
    person_abilities = {}
    for _, row in df.iterrows():
        p_n = np.clip(row[1:].mean(), 0.0001, 0.9999)  # Clip to avoid 0 or 1
        beta_n = np.log(p_n / (1 - p_n))
        person_abilities[row["Student"]] = beta_n
    
    return item_difficulties, person_abilities

def plot_rasch_curve(ability, difficulties, student_name, responses):
    abilities_range = np.linspace(-4, 4, 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main Rasch curve
    ax.plot(abilities_range, expit(abilities_range), 
            color='gray', linestyle='-', linewidth=2, label='Rasch Curve')
    
    # Student ability line
    ax.axvline(x=ability, color='red', linestyle=':', alpha=0.7, 
               linewidth=1.5, label=f"{student_name}'s Ability (β={ability:.2f})")
    
    # Item markers and probabilities
    for i, (item, delta) in enumerate(difficulties.items()):
        prob = expit(ability - delta)
        marker_color = 'green' if responses.iloc[i+1] == 1 else 'red'
        item_label = item.split(':')[0] if ':' in item else item
        ax.plot(delta, prob, 'o', markersize=10, color=marker_color,
                markeredgecolor='black', label=f"{item_label} (δ={delta:.2f})")
    
    # Formatting
    ax.set_xlabel("Person Location (logits)", fontsize=12)
    ax.set_ylabel("Probability of Correct Response", fontsize=12)
    ax.set_title(f"Rasch Model Probability Curve for {student_name}", fontsize=14, pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-4.1, 4.1)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    st.pyplot(fig)

def calculate_improvement_stats(student_score, total_possible):
    current_percent = (student_score / total_possible) * 100
    needed_points = total_possible - student_score
    
    # Calculate Z-score for current performance
    mean_score = total_possible * 0.7  # Assuming average is 70% of total
    std_dev = total_possible * 0.15    # Assuming standard deviation of 15% of total
    z_score = (student_score - mean_score) / std_dev
    
    # Calculate percentile
    percentile = norm.cdf(z_score) * 100
    
    # Calculate probability of reaching 100%
    prob_reach_100 = norm.sf((total_possible - mean_score) / std_dev) * 100
    
    return {
        "current_percent": current_percent,
        "needed_points": needed_points,
        "z_score": z_score,
        "percentile": percentile,
        "prob_reach_100": prob_reach_100
    }

def main():
    st.title("📊 Enhanced Rasch Model Visualization Tool")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df["Total Correct"] = df.iloc[:, 1:].sum(axis=1)
    total_questions = len(df.columns) - 2  # Exclude Student and Total Correct columns
    
    # Calculate Rasch parameters
    item_difficulties, person_abilities = calculate_rasch_parameters(df)
    
    # Layout columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Student Response Data")
        
        # Create styled DataFrame
        styled_df = df.copy()
        for col in styled_df.columns[1:-1]:
            styled_df[col] = styled_df[col].apply(lambda x: "✔️" if x == 1 else "✖️")
        
        st.dataframe(styled_df, height=300)
        
        st.subheader("Model Parameters")
        
        # Item difficulties table
        st.markdown("**Item Difficulties (δ)**")
        item_df = pd.DataFrame.from_dict(item_difficulties, orient='index', 
                                       columns=['Difficulty'])
        st.dataframe(item_df.style.format("{:.3f}"), height=200)
        
        # Person abilities table
        st.markdown("**Student Abilities (β)**")
        person_df = pd.DataFrame.from_dict(person_abilities, orient='index', 
                                         columns=['Ability'])
        st.dataframe(person_df.style.format("{:.3f}"), height=200)
    
    with col2:
        st.subheader("Performance Analysis")
        
        # Select student
        selected_student = st.selectbox("Select Student", df["Student"])
        
        if selected_student:
            ability = person_abilities[selected_student]
            responses = df[df["Student"] == selected_student].iloc[0]
            current_score = responses["Total Correct"]
            
            # Calculate probabilities
            prob_data = []
            for i, (item, delta) in enumerate(item_difficulties.items()):
                prob = expit(ability - delta)
                item_parts = item.split(':')
                item_name = item_parts[0] if len(item_parts) > 1 else item
                question_text = item_parts[1].strip() if len(item_parts) > 1 else ""
                
                prob_data.append({
                    "Item": item_name,
                    "Question": question_text,
                    "Difficulty (δ)": delta,
                    "P(X=1)": prob,
                    "Actual Response": "Correct" if responses.iloc[i+1] == 1 else "Incorrect",
                    "Fit": "✓" if (prob > 0.5) == (responses.iloc[i+1] == 1) else "✗"
                })
            
            prob_df = pd.DataFrame(prob_data)
            
            # Display probability table
            st.dataframe(
                prob_df.style
                .format({"Difficulty (δ)": "{:.3f}", "P(X=1)": "{:.3f}"})
                .apply(lambda x: ['background: #e6ffe6' if x["Fit"] == "✓" 
                                else 'background: #ffe6e6' for i, x in prob_df.iterrows()], 
                      axis=1),
                height=300
            )
            
            # Plot the curve
            plot_rasch_curve(ability, item_difficulties, selected_student, responses)
            
            # Improvement statistics
            st.subheader("Improvement Analysis")
            
            stats = calculate_improvement_stats(current_score, total_questions)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Current Score", f"{current_score}/{total_questions}", 
                          f"{stats['current_percent']:.1f}%")
            
            with col_b:
                st.metric("Points Needed for 100%", stats['needed_points'])
            
            with col_c:
                st.metric("Percentile Rank", f"{stats['percentile']:.1f}%")
            
            # Improvement recommendations
            st.subheader("Improvement Recommendations")
            
            # Identify easiest missed questions
            missed_questions = []
            for i, (item, delta) in enumerate(item_difficulties.items()):
                if responses.iloc[i+1] == 0:  # If question was missed
                    missed_questions.append({
                        "Question": item,
                        "Difficulty": delta,
                        "Probability": expit(ability - delta)
                    })
            
            # Sort by easiest first (highest probability of success)
            missed_questions.sort(key=lambda x: -x["Probability"])
            
            if missed_questions:
                st.write("**Focus on these questions to improve your score:**")
                for i, q in enumerate(missed_questions[:3]):  # Show top 3 recommendations
                    st.write(f"{i+1}. **{q['Question']}** (Difficulty: {q['Difficulty']:.2f}, "
                            f"Your success probability: {q['Probability']*100:.1f}%)")
            else:
                st.success("Congratulations! You've answered all questions correctly.")
            
            # Probability of reaching 100%
            st.write(f"**Probability of reaching 100% based on current performance:** "
                   f"{stats['prob_reach_100']:.2f}%")

if __name__ == "__main__":
    main()



