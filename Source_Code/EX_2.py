import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(base_dir, file_name):
    file_path = os.path.join(base_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist. Please check.")
        exit(1)
    try:
        data = pd.read_csv(file_path, na_values=["N/A"])
        print(f"Loaded {file_path} with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

def preprocess_data(data):
    non_numeric_columns = ["Player", "Nation", "Team", "Position"]
    numeric_columns = [col for col in data.columns if col not in non_numeric_columns]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    return data, numeric_columns

def calculate_rankings(data, numeric_columns):
    rankings = {}
    for col in numeric_columns:
        top_three_high = data[["Player", "Team", col]].sort_values(by=col, ascending=False).head(3)
        top_three_high = top_three_high.rename(columns={col: "Value"})
        top_three_high["Rank"] = ["1st", "2nd", "3rd"]
        
        if data[col].eq(0).all():
            top_three_low = data[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3)
        else:
            non_zero_data = data[data[col] > 0]
            top_three_low = non_zero_data[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3)
        top_three_low = top_three_low.rename(columns={col: "Value"})
        top_three_low["Rank"] = ["1st", "2nd", "3rd"]
        
        rankings[col] = {"Highest": top_three_high, "Lowest": top_three_low}
    return rankings

def save_rankings(base_dir, rankings):
    top_three_path = os.path.join(base_dir, "top_3.txt")
    try:
        with open(top_three_path, "w", encoding="utf-8") as f:
            for stat, data in rankings.items():
                f.write(f"\nStatistic: {stat}\n")
                f.write("\nTop 3 Highest:\n")
                f.write(data["Highest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
                f.write("\n\nTop 3 Lowest:\n")
                f.write(data["Lowest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
                f.write("\n" + "-" * 50 + "\n")
        print(f"Saved top 3 rankings to {top_three_path}")
    except Exception as e:
        print(f"Error saving top 3 rankings: {e}")

def generate_stats(data, numeric_columns):
    stat_rows = []
    league_stats = {"": "all"}
    for col in numeric_columns:
        league_stats[f"Median of {col}"] = data[col].median()
        league_stats[f"Mean of {col}"] = data[col].mean()
        league_stats[f"Std of {col}"] = data[col].std()
    stat_rows.append(league_stats)
    
    teams = sorted(data["Team"].unique())
    for team in teams:
        team_subset = data[data["Team"] == team]
        team_metrics = {"": team}
        for col in numeric_columns:
            team_metrics[f"Median of {col}"] = team_subset[col].median()
            team_metrics[f"Mean of {col}"] = team_subset[col].mean()
            team_metrics[f"Std of {col}"] = team_subset[col].std()
        stat_rows.append(team_metrics)
    
    stats_summary = pd.DataFrame(stat_rows)
    stats_summary = stats_summary.rename(columns={"": ""})
    for col in stats_summary.columns:
        if col != "":
            stats_summary[col] = stats_summary[col].round(2)
    return stats_summary

def save_stats(base_dir, stats_summary):
    stats_csv_path = os.path.join(base_dir, "results2.csv")
    try:
        stats_summary.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved statistics to {stats_csv_path} with {stats_summary.shape[0]} rows and {stats_summary.shape[1]} columns.")
    except Exception as e:
        print(f"Error saving statistics CSV: {e}")

def plot_histograms(data, base_dir, plot_stats):
    histogram_folder = os.path.join(base_dir, "histograms")
    league_histogram_folder = os.path.join(histogram_folder, "league")
    team_histogram_folder = os.path.join(histogram_folder, "teams")
    os.makedirs(league_histogram_folder, exist_ok=True)
    os.makedirs(team_histogram_folder, exist_ok=True)
    
    teams = sorted(data["Team"].unique())
    for stat in plot_stats:
        if stat not in data.columns:
            print(f"Statistic {stat} not found in DataFrame. Skipping...")
            continue
        
        # League-wide histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data[stat], bins=20, color="skyblue", edgecolor="black")
        plt.title(f"League-Wide Distribution of {stat}")
        plt.xlabel(stat)
        plt.ylabel("Number of Players")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(league_histogram_folder, f"{stat}_league.png"), bbox_inches="tight")
        plt.close()
        print(f"Saved league-wide histogram for {stat}")
        
        # Team histograms
        for team in teams:
            team_subset = data[data["Team"] == team]
            plt.figure(figsize=(8, 6))
            plt.hist(team_subset[stat], bins=10, 
                     color="lightgreen" if stat in ["GA90", "TklW", "Blocks"] else "skyblue",
                     edgecolor="black", alpha=0.7)
            plt.title(f"{team} - Distribution of {stat}")
            plt.xlabel(stat)
            plt.ylabel("Number of Players")
            plt.grid(True, alpha=0.3)
            stat_file_name = stat.replace(" ", "_")
            plt.savefig(os.path.join(team_histogram_folder, f"{team}_{stat_file_name}.png"), bbox_inches="tight")
            plt.close()
            print(f"Saved histogram for {team} - {stat}")
    
    print("All histograms for selected statistics have been generated and saved under 'histograms'.")

def analyze_teams(data, numeric_columns, base_dir):
    team_averages = data.groupby("Team")[numeric_columns].mean().reset_index()
    top_team_stats = []
    for stat in numeric_columns:
        if stat not in data.columns:
            print(f"Statistic {stat} not found in DataFrame. Skipping...")
            continue
        top_row = team_averages.loc[team_averages[stat].idxmax()]
        top_team_stats.append({
            "Statistic": stat,
            "Team": top_row["Team"],
            "Mean Value": round(top_row[stat], 2)
        })
    
    top_team_data = pd.DataFrame(top_team_stats)
    top_team_csv_path = os.path.join(base_dir, "highest_team_stats.csv")
    try:
        top_team_data.to_csv(top_team_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved highest team stats to {top_team_csv_path} with {top_team_data.shape[0]} rows.")
    except Exception as e:
        print(f"Error saving highest team stats CSV: {e}")
    
    negative_metrics = ["GA90", "CrdY", "CrdR", "Lost", "Mis", "Dis", "Fls", "Off", "Aerl Lost"]
    positive_stats_data = top_team_data[~top_team_data["Statistic"].isin(negative_metrics)]
    team_rank_counts = positive_stats_data["Team"].value_counts()
    top_team = team_rank_counts.idxmax()
    lead_count = team_rank_counts.max()
    print(f"The best-performing team in the 2024-2025 Premier League season is: {top_team}")
    print(f"They lead in {lead_count} out of {len(positive_stats_data)} positive statistics.")

def main():
    base_dir = r"C:\Users\Admin\OneDrive\Documents\Python\BTLpythonnnn"
    plot_stats = ["Gls per 90", "xG per 90", "SCA90", "GA90", "TklW", "Blocks"]
    
    # Load and preprocess data
    data = load_data(base_dir, "result.csv")
    data, numeric_columns = preprocess_data(data)
    
    # Calculate and save rankings
    rankings = calculate_rankings(data, numeric_columns)
    save_rankings(base_dir, rankings)
    
    # Generate and save statistics
    stats_summary = generate_stats(data, numeric_columns)
    save_stats(base_dir, stats_summary)
    
    # Plot histograms
    plot_histograms(data, base_dir, plot_stats)
    
    # Analyze teams
    analyze_teams(data, numeric_columns, base_dir)

if __name__ == "__main__":
    main()