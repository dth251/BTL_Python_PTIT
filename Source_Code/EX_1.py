import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO
import os

# Kiểm tra thư viện
try:
    import pandas
    import bs4
    import selenium
    import webdriver_manager
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pandas beautifulsoup4 selenium webdriver-manager")
    exit(1)

def parse_age_to_decimal(age_str):
    try:
        if pd.isna(age_str) or age_str == "N/A":
            return "N/A"
        age_str = str(age_str).strip()
        if "-" in age_str:
            years, days = map(int, age_str.split("-"))
            decimal_age = years + (days / 365)
            return round(decimal_age, 2)
        if "." in age_str:
            return round(float(age_str), 2)
        if age_str.isdigit():
            return round(float(age_str), 2)
        return "N/A"
    except (ValueError, AttributeError) as e:
        print(f"Age conversion error for '{age_str}': {e}")
        return "N/A"

def get_country_code(nation_str):
    try:
        if pd.isna(nation_str) or nation_str == "N/A":
            return "N/A"
        return nation_str.split()[-1]
    except (AttributeError, IndexError):
        return "N/A"

def format_player_name(name):
    try:
        if pd.isna(name) or name == "N/A":
            return "N/A"
        if "," in name:
            parts = [part.strip() for part in name.split(",")]
            if len(parts) >= 2:
                return " ".join(parts[::-1])
        return " ".join(name.split()).strip()
    except (AttributeError, TypeError):
        return "N/A"

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver
    except Exception as e:
        print(f"Error initializing ChromeDriver: {e}")
        exit(1)

def scrape_table(driver, url, table_id):
    print(f"Processing {table_id} from {url}")
    driver.get(url)
    time.sleep(3)
    page_soup = BeautifulSoup(driver.page_source, "html.parser")
    html_comments = page_soup.find_all(string=lambda text: isinstance(text, Comment))
    data_table = None
    for comment in html_comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            data_table = comment_soup.find("table", {"id": table_id})
            if data_table:
                break
    if not data_table:
        print(f"Table {table_id} not found!")
        return None
    try:
        table_data = pd.read_html(StringIO(str(data_table)), header=0)[0]
        return table_data
    except Exception as e:
        print(f"Error reading table {table_id}: {e}")
        return None

def process_table(table_data, table_id, rename_map):
    if table_data is None:
        return None
    print(f"Original columns in {table_id}:", table_data.columns.tolist())
    table_data = table_data.rename(columns=rename_map.get(table_id, {}))
    table_data = table_data.loc[:, ~table_data.columns.duplicated()]
    if "Player" in table_data.columns:
        table_data["Player"] = table_data["Player"].apply(format_player_name)
        print(f"Sample Player names in {table_id}:", table_data["Player"].head(5).tolist())
    if "Age" in table_data.columns:
        print(f"Raw Age values in {table_id} (before conversion):", table_data["Age"].head(5).tolist())
        table_data["Age"] = table_data["Age"].apply(parse_age_to_decimal)
        print(f"Processed Age values in {table_id} (after conversion):", table_data["Age"].head(5).tolist())
    print(f"Renamed and cleaned columns in {table_id}:", table_data.columns.tolist())
    return table_data

def merge_tables(tables, target_columns):
    final_data = None
    for table_id, table_data in tables.items():
        if table_data is None:
            continue
        table_data = table_data[[col for col in table_data.columns if col in target_columns]]
        table_data = table_data.drop_duplicates(subset=["Player"], keep="first")
        if final_data is None:
            final_data = table_data
        else:
            try:
                final_data = pd.merge(final_data, table_data, on="Player", how="outer", validate="1:1")
            except Exception as e:
                print(f"Merge error for {table_id}: {e}")
                continue
    return final_data

def finalize_data(data, target_columns):
    if data is None:
        print("No data to process.")
        return None
    data = data.loc[:, [col for col in target_columns if col in data.columns]]
    data["Minutes"] = pd.to_numeric(data["Minutes"], errors="coerce")
    
    integer_columns = ["Matches Played", "Starts", "Minutes", "Gls", "Ast", "crdY", "crdR", "PrgC", "PrgP", "PrgR",
                       "Cmp", "TotDist", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", "Sh", "Pass", "Int",
                       "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Take-Ons Att",
                       "Carries", "Carries 1_3", "CPA", "Mis", "Dis", "Rec", "Rec PrgR",
                       "Fls", "Fld", "Off", "Crs", "Recov", "Aerl Won", "Aerl Lost"]
    decimal_columns = ["Age", "xG", "xAG", "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90", "GA90", "Save%", "CS%", "PK Save%",
                       "SoT%", "SoT per 90", "G per Sh", "Dist", "Cmp%", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA",
                       "CrsPA", "SCA", "SCA90", "GCA", "GCA90", "Succ%", "Tkld%", "ProDist", "Aerl Won%"]
    text_columns = ["Player", "Nation", "Team", "Position"]
    
    for col in integer_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")
    for col in decimal_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").round(2)
    data = data[data["Minutes"].notna() & (data["Minutes"] > 90)]
    
    if "Nation" in data.columns:
        data["Nation"] = data["Nation"].apply(get_country_code)
    if "Player" in data.columns:
        data["Player"] = data["Player"].apply(format_player_name)
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].fillna("N/A")
    
    return data

def save_data(data, output_path):
    if data is None or data.empty:
        print("No data to save.")
        return
    try:
        print("\nPreview of final DataFrame (first 5 rows) before saving to result.csv:")
        print(data.head(5).to_string())
        data.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="N/A")
        print(f"Successfully saved merged data to {output_path} with {data.shape[0]} rows and {data.shape[1]} columns.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def main():
    base_dir = r"C:\Users\Admin\OneDrive\Documents\Python\BTLpythonnnn"
    output_csv_path = os.path.join(base_dir, "result.csv")
    
    stat_urls = [
        "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats",
        "https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats",
    ]
    table_identifiers = [
        "stats_standard",
        "stats_keeper",
        "stats_shooting",
        "stats_passing",
        "stats_gca",
        "stats_defense",
        "stats_possession",
        "stats_misc",
    ]
    target_columns = [
        "Player", "Nation", "Team", "Position", "Age",
        "Matches Played", "Starts", "Minutes",
        "Gls", "Ast", "crdY", "crdR",
        "xG", "xAG",
        "PrgC", "PrgP", "PrgR",
        "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
        "GA90", "Save%", "CS%", "PK Save%",
        "SoT%", "SoT per 90", "G per Sh", "Dist",
        "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA", "CrsPA",
        "SCA", "SCA90", "GCA", "GCA90",
        "Tkl", "TklW",
        "Deff Att", "Lost",
        "Blocks", "Sh", "Pass", "Int",
        "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
        "Take-Ons Att", "Succ%", "Tkld%",
        "Carries", "ProDist", "Carries 1_3", "CPA", "Mis", "Dis",
        "Rec", "Rec PrgR",
        "Fls", "Fld", "Off", "Crs", "Recov",
        "Aerl Won", "Aerl Lost", "Aerl Won%"
    ]
    rename_columns_map = {
        "stats_standard": {
            "Unnamed: 1": "Player",
            "Unnamed: 2": "Nation",
            "Unnamed: 3": "Position",
            "Unnamed: 4": "Team",
            "Unnamed: 5": "Age",
            "Playing Time": "Matches Played",
            "Playing Time.1": "Starts",
            "Playing Time.2": "Minutes",
            "Performance": "Gls",
            "Performance.1": "Ast",
            "Performance.6": "crdY",
            "Performance.7": "crdR",
            "Expected": "xG",
            "Expected.2": "xAG",
            "Progression": "PrgC",
            "Progression.1": "PrgP",
            "Progression.2": "PrgR",
            "Per 90 Minutes": "Gls per 90",
            "Per 90 Minutes.1": "Ast per 90",
            "Per 90 Minutes.5": "xG per 90",
            "Per 90 Minutes.6": "xAG per 90"
        },
        "stats_keeper": {
            "Unnamed: 1": "Player",
            "Performance.1": "GA90",
            "Performance.4": "Save%",
            "Performance.9": "CS%",
            "Penalty Kicks.4": "PK Save%"
        },
        "stats_shooting": {
            "Unnamed: 1": "Player",
            "Standard.3": "SoT%",
            "Standard.5": "SoT per 90",
            "Standard.6": "G per Sh",
            "Standard.8": "Dist"
        },
        "stats_passing": {
            "Unnamed: 1": "Player",
            "Total": "Cmp",
            "Total.2": "Cmp%",
            "Total.3": "TotDist",
            "Short.2": "ShortCmp%",
            "Medium.2": "MedCmp%",
            "Long.2": "LongCmp%",
            "Unnamed: 26": "KP",
            "Unnamed: 27": "Pass into 1_3",
            "Unnamed: 28": "PPA",
            "Unnamed: 29": "CrsPA",
        },
        "stats_gca": {
            "Unnamed: 1": "Player",
            "SCA.1": "SCA90",
            "GCA.1": "GCA90",
        },
        "stats_defense": {
            "Unnamed: 1": "Player",
            "Tackles": "Tkl", "Tackles.1": "TklW",
            "Challenges.1": "Deff Att",
            "Challenges.3": "Lost",
            "Blocks": "Blocks",
            "Blocks.1": "Sh",
            "Blocks.2": "Pass",
            "Unnamed: 20": "Int",
        },
        "stats_possession": {
            "Unnamed: 1": "Player",
            "Touches": "Touches",
            "Touches.1": "Def Pen",
            "Touches.2": "Def 3rd",
            "Touches.3": "Mid 3rd",
            "Touches.4": "Att 3rd",
            "Touches.5": "Att Pen",
            "Touches.6": "Live",
            "Take-Ons": "Take-Ons Att",
            "Take-Ons.2": "Succ%",
            "Take-Ons.4": "Tkld%",
            "Carries": "Carries",
            "Carries.2": "ProDist",
            "Carries.4": "Carries 1_3",
            "Carries.5": "CPA",
            "Carries.6": "Mis",
            "Carries.7": "Dis",
            "Receiving": "Rec",
            "Receiving.1": "Rec PrgR",
        },
        "stats_misc": {
            "Unnamed: 1": "Player",
            "Performance.3": "Fls",
            "Performance.4": "Fld",
            "Performance.5": "Off",
            "Performance.6": "Crs",
            "Performance.12": "Recov",
            "Aerial Duels": "Aerl Won",
            "Aerial Duels.1": "Aerl Lost",
            "Aerial Duels.2": "Aerl Won%"
        }
    }
    
    # Thiết lập thư mục và kiểm tra
    os.makedirs(base_dir, exist_ok=True)
    driver = setup_driver()
    try:
        # Cào và xử lý từng bảng
        tables = {}
        for url, table_id in zip(stat_urls, table_identifiers):
            table_data = scrape_table(driver, url, table_id)
            table_data = process_table(table_data, table_id, rename_columns_map)
            if table_data is not None:
                tables[table_id] = table_data
        
        # Gộp và hoàn thiện dữ liệu
        final_data = merge_tables(tables, target_columns)
        final_data = finalize_data(final_data, target_columns)
        
        # Lưu dữ liệu
        save_data(final_data, output_csv_path)
    finally:
        driver.quit()
        print("WebDriver closed.")

if __name__ == "__main__":
    main()