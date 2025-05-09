import pandas as pd
import numpy as np
import os
import re
import time
import logging
from fuzzywuzzy import fuzz, process
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Kiểm tra thư viện
try:
    import pandas
    import fuzzywuzzy
    import selenium
    import webdriver_manager
    import sklearn
except ImportError as e:
    logging.error(f"Thiếu thư viện: {e}. Cài đặt bằng lệnh: pip install pandas fuzzywuzzy python-Levenshtein selenium webdriver-manager scikit-learn")
    exit(1)

# Cấu hình cho các vị trí cầu thủ
roles_config = {
    'Goalkeeper': {
        'role_filter': 'GK',
        'attributes': [
            'Save%', 'CS%', 'GA90', 'Minutes', 'Age', 'PK Save%', 'Team', 'Nation'
        ],
        'key_attributes': ['Save%', 'CS%', 'PK Save%']
    },
    'Defender': {
        'role_filter': 'DF',
        'attributes': [
            'Tkl', 'TklW', 'Int', 'Blocks', 'Recov', 'Minutes', 'Team', 'Age', 'Nation', 'Aerl Won%',
            'Aerl Won', 'Cmp', 'Cmp%', 'PrgP', 'LongCmp%', 'Carries', 'Touches', 'Dis', 'Mis'
        ],
        'key_attributes': ['Tkl', 'TklW', 'Int', 'Blocks', 'Aerl Won%', 'Aerl Won', 'Recov']
    },
    'Midfielder': {
        'role_filter': 'MF',
        'attributes': [
            'Cmp%', 'KP', 'PPA', 'PrgP', 'Tkl', 'Ast', 'SCA', 'Touches', 'Minutes', 'Team', 'Age', 'Nation',
            'Pass into 1_3', 'xAG', 'Carries 1_3', 'ProDist', 'Rec', 'Mis', 'Dis'
        ],
        'key_attributes': ['KP', 'PPA', 'PrgP', 'SCA', 'xAG', 'Pass into 1_3', 'Carries 1_3']
    },
    'Forward': {
        'role_filter': 'FW',
        'attributes': [
            'Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SoT%', 'G per Sh', 'SCA90', 'GCA90',
            'PrgC', 'Carries 1_3', 'Aerl Won%', 'Team', 'Age', 'Minutes'
        ],
        'key_attributes': ['Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SCA90', 'GCA90']
    }
}

class TransferScraper:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.input_file = os.path.join(data_dir, "result.csv")
        self.filtered_file = os.path.join(data_dir, "players_over_900_minutes.csv")
        self.output_file = os.path.join(data_dir, "player_transfer_fee.csv")
        self.base_url = "https://www.footballtransfers.com/us/transfers/confirmed/2024-2025/uk-premier-league/"
        os.makedirs(data_dir, exist_ok=True)

    def validate_input(self):
        if not os.path.exists(self.input_file):
            logging.error(f"Input file {self.input_file} not found.")
            exit()

    def filter_players(self):
        try:
            df = pd.read_csv(self.input_file, na_values=["N/A"])
            filtered_df = df[df['Minutes'] > 900].copy()
            logging.info(f"Found {len(filtered_df)} players with over 900 minutes.")
            filtered_df.to_csv(self.filtered_file, index=False, encoding='utf-8-sig')
            logging.info(f"Saved {filtered_df.shape[0]} players to {self.filtered_file}.")
            return filtered_df
        except Exception as e:
            logging.error(f"Error reading or processing {self.input_file}: {str(e)}")
            exit()

    def load_filtered_players(self):
        try:
            df = pd.read_csv(self.filtered_file)
            names = df['Player'].str.strip()
            short_names = [self._shorten_name(name) for name in names]
            minutes_dict = dict(zip(names, df['Minutes']))
            return short_names, minutes_dict
        except Exception as e:
            logging.error(f"Error reading {self.filtered_file}: {str(e)}")
            exit()

    def _shorten_name(self, name):
        parts = name.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else name

    def setup_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def scrape_transfers(self, short_names):
        driver = self.setup_driver()
        urls = [f"{self.base_url}{i}" for i in range(1, 15)]
        results = []

        try:
            for url in urls:
                logging.info(f"Accessing {url}")
                driver.get(url)
                try:
                    table = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "transfer-table"))
                    )
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    logging.info(f"Found {len(rows)} rows at {url}")
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if len(cols) >= 2:
                            player = cols[0].text.strip().split("\n")[0].strip()
                            short_player = self._shorten_name(player)
                            price = cols[-1].text.strip() if len(cols) >= 3 else "N/A"
                            logging.info(f"Processing: {player}, Short: {short_player}, Price: {price}")
                            match = process.extractOne(short_player, short_names, scorer=fuzz.token_sort_ratio)
                            if match and match[1] >= 80:
                                logging.info(f"Matched: {player} -> {match[0]} (Score: {match[1]})")
                                results.append([player, price])
                        else:
                            logging.info(f"Skipping row with {len(cols)} columns")
                except Exception as e:
                    logging.error(f"Error at {url}: {str(e)}")
        finally:
            driver.quit()

        return results

    def save_results(self, results):
        if results:
            result_df = pd.DataFrame(results, columns=['Player', 'Price'])
            result_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            logging.info(f"Saved {len(results)} records to {self.output_file}")
        else:
            logging.warning("No matching players found.")

    def run(self):
        self.validate_input()
        self.filter_players()
        short_names, _ = self.load_filtered_players()
        results = self.scrape_transfers(short_names)
        self.save_results(results)

class ETVScraper:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_csv = os.path.join(root_dir, "result.csv")
        self.output_csv = os.path.join(root_dir, "all_estimate_transfer_fee.csv")
        self.base_url = "https://www.footballtransfers.com/us/players/uk-premier-league/"
        os.makedirs(root_dir, exist_ok=True)
        self.driver = None
        self.max_retries = 3

    def _shorten_name(self, name):
        special_cases = {
            "Manuel Ugarte Ribeiro": "Manuel Ugarte",
            "Igor Júlio": "Igor",
            "Igor Thiago": "Thiago",
            "Felipe Morato": "Morato",
            "Nathan Wood-Gordon": "Nathan Wood",
            "Bobby Reid": "Bobby Cordova-Reid",
            "J. Philogene": "Jaden Philogene Bidace"
        }
        if name in special_cases:
            return special_cases[name]
        parts = name.strip().split()
        return f"{parts[0]} {parts[-1]}" if len(parts) >= 3 else name

    def load_players(self):
        player_list = []
        position_map = {}
        original_name_map = {}
        if os.path.exists(self.input_csv):
            try:
                df = pd.read_csv(self.input_csv, encoding='utf-8-sig')
                player_list = df['Player'].str.strip().apply(self._shorten_name).tolist()
                position_map = dict(zip(player_list, df['Position']))
                original_name_map = dict(zip(player_list, df['Player'].str.strip()))
                logging.info(f"Đã tải {len(player_list)} cầu thủ từ result.csv")
            except Exception as e:
                logging.error(f"Lỗi khi đọc result.csv: {e}")
                logging.info("Tiếp tục mà không so khớp cầu thủ...")
        else:
            logging.warning("Không tìm thấy result.csv. Cào tất cả cầu thủ mà không so khớp.")
        return player_list, position_map, original_name_map

    def setup_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        for attempt in range(self.max_retries):
            try:
                self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                logging.info("Khởi tạo ChromeDriver thành công")
                return
            except WebDriverException as e:
                logging.error(f"Lỗi khởi tạo ChromeDriver (lần thử {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)
        logging.error("Không thể khởi tạo ChromeDriver sau các lần thử")
        exit(1)

    def get_max_pages(self):
        for attempt in range(self.max_retries):
            try:
                self.driver.get(self.base_url + "1")
                pagination = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "pagination"))
                )
                page_links = pagination.find_elements(By.TAG_NAME, "a")
                max_pages = 1
                for link in page_links:
                    try:
                        page_num = int(link.text)
                        max_pages = max(max_pages, page_num)
                    except ValueError:
                        continue
                logging.info(f"Phát hiện {max_pages} trang để cào")
                return max_pages
            except (TimeoutException, NoSuchElementException) as e:
                logging.error(f"Lỗi phát hiện phân trang (lần thử {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)
        logging.warning("Mặc định dùng 22 trang do không phát hiện được phân trang")
        return 22

    def scrape_page(self, url, player_list, position_map, original_name_map):
        gk_players = []
        df_players = []
        mf_players = []
        fw_players = []
        unmatched_players = []
        
        for attempt in range(self.max_retries):
            try:
                self.driver.get(url)
                table = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "similar-players-table"))
                )
                rows = table.find_elements(By.TAG_NAME, "tr")
                logging.info(f"Tìm thấy {len(rows)} hàng tại {url}")
                
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if cols and len(cols) >= 2:
                        player_name = cols[1].text.strip().split("\n")[0].strip()
                        short_name = self._shorten_name(player_name)
                        etv = cols[-1].text.strip() if len(cols) >= 3 else "N/A"
                        
                        if player_list:
                            best_match = process.extractOne(short_name, player_list, scorer=fuzz.token_sort_ratio)
                            if best_match and best_match[1] >= 70:
                                matched_name = best_match[0]
                                original_name = original_name_map.get(matched_name, matched_name)
                                position = position_map.get(matched_name, "Unknown")
                                logging.info(f"Tìm thấy khớp: {player_name} -> {original_name} (điểm: {best_match[1]}, Vị trí: {position})")
                                
                                if "GK" in position:
                                    gk_players.append([original_name, position, etv])
                                elif position.startswith("DF"):
                                    df_players.append([original_name, position, etv])
                                elif position.startswith("MF"):
                                    mf_players.append([original_name, position, etv])
                                elif position.startswith("FW"):
                                    fw_players.append([original_name, position, etv])
                            else:
                                logging.info(f"Không khớp: {player_name} (khớp tốt nhất: {best_match[0] if best_match else 'None'}, điểm: {best_match[1] if best_match else 'N/A'})")
                                unmatched_players.append([player_name, "Unknown", etv])
                        else:
                            unmatched_players.append([player_name, "Unknown", etv])
                return gk_players, df_players, mf_players, fw_players, unmatched_players
            except (TimeoutException, NoSuchElementException) as e:
                logging.error(f"Lỗi xử lý {url} (lần thử {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)
        logging.error(f"Không thể xử lý {url} sau {self.max_retries} lần thử")
        return gk_players, df_players, mf_players, fw_players, unmatched_players

    def save_output(self, gk_players, df_players, mf_players, fw_players, unmatched_players, player_list):
        all_players = gk_players + df_players + mf_players + fw_players
        if not player_list:
            all_players += unmatched_players
        
        if all_players:
            try:
                df = pd.DataFrame(all_players, columns=['Player', 'Position', 'Price'])
                df = df.drop_duplicates(subset=['Player'])
                df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
                logging.info(f"Tệp 'all_estimate_transfer_fee.csv' đã được lưu tại: {self.output_csv}")
                logging.info(f"Tổng số cầu thủ cào được: {len(df)} (GK: {len(gk_players)}, DF: {len(df_players)}, MF: {len(mf_players)}, FW: {len(fw_players)})")
                if unmatched_players and player_list:
                    logging.info(f"Cầu thủ không khớp: {len(unmatched_players)}")
            except Exception as e:
                logging.error(f"Lỗi khi lưu tệp CSV: {e}")
        else:
            logging.warning("Không tìm thấy cầu thủ nào. Kiểm tra URL, cấu trúc bảng, hoặc kết nối mạng.")

    def run(self):
        player_list, position_map, original_name_map = self.load_players()
        self.setup_driver()
        try:
            max_pages = self.get_max_pages()
            urls = [f"{self.base_url}{i}" for i in range(1, max_pages + 1)]
            gk_players, df_players, mf_players, fw_players, unmatched_players = [], [], [], [], []
            
            for url in urls:
                logging.info(f"Đang cào: {url}")
                gk, df, mf, fw, unmatched = self.scrape_page(url, player_list, position_map, original_name_map)
                gk_players.extend(gk)
                df_players.extend(df)
                mf_players.extend(mf)
                fw_players.extend(fw)
                unmatched_players.extend(unmatched)
            
            self.save_output(gk_players, df_players, mf_players, fw_players, unmatched_players, player_list)
        except Exception as e:
            logging.error(f"Lỗi bất ngờ khi cào dữ liệu: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                logging.info("Đã đóng WebDriver.")

class TransferValuePredictor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.stats_path = os.path.join(base_dir, "result.csv")
        self.valuation_path = os.path.join(base_dir, "all_estimate_transfer_fee.csv")
        self.output_path = os.path.join(base_dir, "ml_estimated_values_linear.csv")
        os.makedirs(base_dir, exist_ok=True)

    def simplify_name(self, name):
        if not isinstance(name, str):
            return ""
        parts = name.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else name

    def convert_valuation(self, val_text):
        if pd.isna(val_text) or val_text in ["N/A", ""]:
            return np.nan
        try:
            val_text = re.sub(r'[€£]', '', val_text).strip().upper()
            multiplier = 1000000 if 'M' in val_text else 1000 if 'K' in val_text else 1
            value = float(re.sub(r'[MK]', '', val_text)) * multiplier
            return value
        except (ValueError, TypeError):
            return np.nan

    def match_player_name(self, name, options, min_score=90):
        if not isinstance(name, str):
            return None, None
        simplified_name = self.simplify_name(name).lower()
        simplified_options = [self.simplify_name(opt).lower() for opt in options if isinstance(opt, str)]
        match = process.extractOne(
            simplified_name,
            simplified_options,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=min_score
        )
        if match is not None:
            matched_idx = simplified_options.index(match[0])
            return options[matched_idx], match[1]
        return None, None

    def load_data(self):
        try:
            stats_data = pd.read_csv(self.stats_path)
            valuation_data = pd.read_csv(self.valuation_path)
            return stats_data, valuation_data
        except FileNotFoundError as e:
            logging.error(f"File not found - {e}")
            return None, None

    def process_role(self, role, config, stats_data, valuation_data):
        stats_data['Main_Role'] = stats_data['Position'].astype(str).str.split(r'[,/]').str[0].str.strip()
        stats_data = stats_data[stats_data['Main_Role'].str.upper() == config['role_filter'].upper()].copy()
        
        player_list = valuation_data['Player'].dropna().tolist()
        
        stats_data['Linked_Name'] = None
        stats_data['Link_Score'] = None
        stats_data['Valuation'] = np.nan
        
        for idx, row in stats_data.iterrows():
            linked_name, score = self.match_player_name(row['Player'], player_list)
            if linked_name:
                stats_data.at[idx, 'Linked_Name'] = linked_name
                stats_data.at[idx, 'Link_Score'] = score
                linked_row = valuation_data[valuation_data['Player'] == linked_name]
                if not linked_row.empty:
                    val_value = self.convert_valuation(linked_row['Price'].iloc[0])
                    stats_data.at[idx, 'Valuation'] = val_value
        
        filtered_data = stats_data[stats_data['Linked_Name'].notna()].copy()
        filtered_data = filtered_data.drop_duplicates(subset='Linked_Name')
        
        unmatched_players = stats_data[stats_data['Linked_Name'].isna()]['Player'].dropna().tolist()
        if unmatched_players:
            logging.info(f"Players in {role} not matched: {len(unmatched_players)} players unmatched.")
            logging.info(unmatched_players)
        
        attributes = config['attributes']
        target_col = 'Valuation'
        
        for col in attributes:
            if col in ['Team', 'Nation']:
                filtered_data[col] = filtered_data[col].fillna('Unknown')
            else:
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
                median_val = filtered_data[col].median()
                filtered_data[col] = filtered_data[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        numeric_attrs = [col for col in attributes if col not in ['Team', 'Nation']]
        for col in numeric_attrs:
            filtered_data[col] = np.log1p(filtered_data[col].clip(lower=0))
        
        for col in config['key_attributes']:
            if col in filtered_data.columns:
                filtered_data[col] = filtered_data[col] * 2.0
        if 'Minutes' in filtered_data.columns:
            filtered_data['Minutes'] = filtered_data['Minutes'] * 1.5
        if 'Age' in filtered_data.columns:
            filtered_data['Age'] = filtered_data['Age'] * 0.5
        
        ml_data = filtered_data.dropna(subset=[target_col]).copy()
        if ml_data.empty:
            logging.error(f"No valid Valuation data for {role}.")
            return None, unmatched_players
        
        return filtered_data, ml_data, attributes, target_col, unmatched_players

    def train_model(self, data, attributes, target_col):
        X = data[attributes]
        y = data[target_col]
        
        numeric_attrs = [col for col in attributes if col not in ['Team', 'Nation']]
        categorical_attrs = [col for col in attributes if col in ['Team', 'Nation']]
        
        if len(data) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            logging.warning(f"Insufficient data for splitting train/test set.")
            X_train, y_train = X, y
            X_test, y_test = X, y
        
        data_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_attrs),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_attrs)
            ])
        
        model_pipeline = Pipeline([
            ('transformer', data_transformer),
            ('model', LinearRegression())
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        if len(X_test) > 0:
            y_pred = model_pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
        
        return model_pipeline

    def save_output(self, role_outputs, unmatched_players):
        if role_outputs:
            final_output = pd.concat(role_outputs, ignore_index=True)
            final_output = final_output.sort_values(by='Predicted_Transfer_Value_M', ascending=False)
            try:
                final_output.to_csv(self.output_path, index=False)
                logging.info(f"Estimated player values saved to '{self.output_path}'")
            except Exception as e:
                logging.error(f"Error saving CSV: {e}")
        
        if unmatched_players:
            logging.info("\nSummary of unmatched players:")
            for role, player in unmatched_players:
                logging.info(f"{role}: {player}")

    def run(self):
        standard_output_columns = [
            'Player', 'Team', 'Nation', 'Position', 'Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'
        ]
        
        stats_data, valuation_data = self.load_data()
        if stats_data is None or valuation_data is None:
            return
        
        role_outputs = []
        unmatched_players = []
        
        for role, config in roles_config.items():
            logging.info(f"\nProcessing {role}...")
            config['data_path'] = self.valuation_path
            filtered_data, ml_data, attributes, target_col, unmatched = self.process_role(role, config, stats_data, valuation_data)
            
            if filtered_data is None:
                continue
            
            model_pipeline = self.train_model(ml_data, attributes, target_col)
            
            filtered_data['Estimated_Value'] = model_pipeline.predict(filtered_data[attributes])
            filtered_data['Estimated_Value'] = filtered_data['Estimated_Value'].clip(lower=100_000, upper=200_000_000)
            filtered_data['Predicted_Transfer_Value_M'] = (filtered_data['Estimated_Value'] / 1_000_000).round(2)
            filtered_data['Actual_Transfer_Value_M'] = (filtered_data['Valuation'] / 1_000_000).round(2)
            
            for col in standard_output_columns:
                if col not in filtered_data.columns:
                    filtered_data[col] = np.nan if col in ['Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'] else ''
            
            filtered_data['Position'] = role
            output_data = filtered_data[standard_output_columns].copy()
            
            numeric_attrs = [col for col in attributes if col not in ['Team', 'Nation']]
            numeric_attrs_no_age = [col for col in numeric_attrs if col != 'Age']
            for col in numeric_attrs_no_age:
                if col in output_data.columns:
                    output_data[col] = np.expm1(output_data[col]).round(2)
            if 'Age' in output_data.columns:
                output_data['Age'] = np.expm1(output_data['Age']).round(0)
                median_age = output_data['Age'].median()
                output_data['Age'] = output_data['Age'].fillna(median_age).astype(int)
            
            role_outputs.append(output_data)
            unmatched_players.extend([(role, player) for player in unmatched])
        
        self.save_output(role_outputs, unmatched_players)

def main():
    base_dir = r"C:\Users\Admin\OneDrive\Documents\Python\BTLpythonnnn"
    
    # Chạy TransferScraper
    transfer_scraper = TransferScraper(base_dir)
    transfer_scraper.run()
    
    # Chạy ETVScraper
    etv_scraper = ETVScraper(base_dir)
    etv_scraper.run()
    
    # Chạy TransferValuePredictor
    predictor = TransferValuePredictor(base_dir)
    predictor.run()

if __name__ == "__main__":
    main()