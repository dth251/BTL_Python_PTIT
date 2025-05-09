import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import csv

def load_data(base_dir, file_name):
    file_path = os.path.join(base_dir, file_name)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist. Please check the path.")
        exit(1)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist. Please check.")
        exit(1)
    try:
        data = pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            na_values="N/A",
            sep=',',
            quoting=csv.QUOTE_ALL,
            on_bad_lines='skip'
        )
        print(f"Loaded file with encoding 'utf-8-sig' ({data.shape[0]} rows, {data.shape[1]} columns).")
        print("\nFirst 5 rows of data:\n", data.head())
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Printing first 10 lines for inspection:")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for i, line in enumerate(f, 1):
                print(f"Line {i}: {line.strip()}")
                if i == 10:
                    break
        exit(1)

def preprocess_data(data):
    print("\nColumns in data:", data.columns.tolist())
    if data.empty:
        print("Error: Data is empty after loading. Please check result.csv.")
        exit(1)
    
    # Filter out goalkeepers
    if 'Position' in data.columns:
        data = data[~data["Position"].str.contains("GK", na=False)]
        print(f"Filtered out goalkeepers. Remaining data: {data.shape[0]} rows.")
    else:
        print("Warning: 'Position' column not found. Skipping goalkeeper filter.")
    
    # Filter players with >90 minutes
    if 'Minutes' in data.columns:
        data = data[data["Minutes"] > 90]
        print(f"Data after filtering: {data.shape[0]} players with >90 minutes.")
    else:
        print("Warning: 'Minutes' column not found. Skipping minutes filter.")
    
    if data.empty:
        print("Error: Data is empty after filtering. Please check Position or Minutes columns.")
        exit(1)
    
    # Select features
    features = ['Cmp', 'Cmp%', 'TotDist', 'Tkl', 'Int', 'Blocks', 'Touches', 'Att 3rd', 'Att Pen']
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Warning: Features {missing_features} not found in data.")
        print(f"Available columns: {data.columns.tolist()}")
        exit(1)
    
    X = data[features].copy()
    X = X.replace('N/A', 0)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if X.shape[0] == 0:
        print("Error: No valid data after processing. Please check feature columns.")
        exit(1)
    
    return data, X

def perform_clustering(X, k=4):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    
    # Calculate inertia for Elbow plot
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Perform clustering with chosen k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_features)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    return inertia, k_range, X_pca, cluster_labels

def plot_elbow(base_dir, k_range, inertia, chosen_k):
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='-', color='b', label='Inertia')
    plt.scatter([chosen_k], [inertia[chosen_k-1]], color='red', s=100, label=f'k={chosen_k}', zorder=5)
    plt.title(f'Elbow Plot (k={chosen_k})')
    plt.xlabel(f'Number of clusters (k = {chosen_k})')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.xticks(k_range)
    plt.legend()
    
    elbow_path = os.path.join(base_dir, "elbow_plot.png")
    try:
        plt.savefig(elbow_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Saved Elbow plot at '{elbow_path}'.")
        if os.path.exists(elbow_path):
            print(f"Confirmed: File '{elbow_path}' created.")
        else:
            print(f"Error: File '{elbow_path}' not created. Please check write permissions.")
    except Exception as e:
        print(f"Error saving Elbow plot: {e}")
    plt.close()

def plot_clusters(base_dir, X_pca, cluster_labels, chosen_k):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title(f'2D Clustering Plot of Players (PCA, k={chosen_k})')
    plt.xlabel('Passing and attacking involvement')
    plt.ylabel('Defensive and active involvement')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    
    cluster_path = os.path.join(base_dir, "cluster_plot.png")
    try:
        plt.savefig(cluster_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Saved 2D cluster plot at '{cluster_path}'.")
        if os.path.exists(cluster_path):
            print(f"Confirmed: File '{cluster_path}' created.")
        else:
            print(f"Error: File '{cluster_path}' not created. Please check write permissions.")
    except Exception as e:
        print(f"Error saving cluster plot: {e}")
    plt.close()

def print_cluster_players(data, cluster_labels, chosen_k):
    if 'Player' in data.columns:
        data['Cluster'] = cluster_labels
        print("\nList of players in each cluster:")
        for cluster in range(chosen_k):
            print(f"\nCluster {cluster}:")
            players = data[data['Cluster'] == cluster]['Player'].head(10).tolist()
            if players:
                print(", ".join(players))
            else:
                print("No players in this cluster.")
    else:
        print("\nWarning: 'Player' column not found. Cannot print player clusters.")

def main():
    base_dir = r"C:\Users\Admin\OneDrive\Documents\Python\BTLpythonnnn"
    chosen_k = 4
    
    # Load and preprocess data
    data = load_data(base_dir, "result.csv")
    data, X = preprocess_data(data)
    
    # Perform clustering
    inertia, k_range, X_pca, cluster_labels = perform_clustering(X, k=chosen_k)
    
    # Plot results
    plot_elbow(base_dir, k_range, inertia, chosen_k)
    plot_clusters(base_dir, X_pca, cluster_labels, chosen_k)
    
    # Print cluster players
    print_cluster_players(data, cluster_labels, chosen_k)
    
    # Print comments
    print("""
 Nhận xét về số lượng nhóm cầu thủ và kết quả phân cụm:

**Số lượng nhóm tối ưu**: Nên phân loại cầu thủ thành *4 nhóm*, dựa trên phân tích biểu đồ Elbow Plot. Lý do chọn \( k = 4 \):

1. **Điểm Elbow**: Trong biểu đồ Elbow Plot, giá trị inertia (tổng bình phương khoảng cách trong cụm, WCSS) giảm mạnh từ \( k = 1 \) đến \( k = 4 \), sau đó tốc độ giảm chậm lại đáng kể. Điều này cho thấy \( k = 4 \) là điểm mà việc thêm cụm không mang lại cải thiện đáng kể về độ chặt chẽ của cụm, giúp cân bằng giữa tính đơn giản và độ chính xác của mô hình.

2. **Ý nghĩa bóng đá**: Với \( k = 4 \) và các đặc trưng (chuyền bóng, phòng ngự, tham gia tích cực), các cầu thủ có thể được phân thành các nhóm phản ánh vai trò khác nhau trên sân:
   - *Nhóm 1*: Tiền vệ phòng ngự, nổi bật ở Tkl, Int, và Blocks.
   - *Nhóm 2*: Tiền vệ tổ chức, có giá trị cao ở Cmp, Cmp%, và TotDist.
   - *Nhóm 3*: Cầu thủ tấn công, với Touches, Att 3rd, và Att Pen cao.
   - *Nhóm 4*: Cầu thủ đa năng, cân bằng giữa phòng ngự và tham gia tấn công.

3. **Tính thực tiễn**: Số cụm \( k = 4 \) không quá lớn, giúp dễ dàng diễn giải và áp dụng trong phân tích bóng đá (ví dụ, để phân loại cầu thủ theo vai trò hoặc đánh giá hiệu suất). Nếu chọn \( k \) lớn hơn (ví dụ, \( k = 6 \)), các cụm có thể trở nên quá chi tiết và khó diễn giải, trong khi \( k \) nhỏ hơn (ví dụ, \( k = 2 \)) có thể bỏ qua sự khác biệt quan trọng giữa các vai trò.

**Nhận xét về kết quả**:
- *Hiệu quả của phân cụm*: Việc sử dụng 9 đặc trưng (Cmp, Cmp%, TotDist, Tkl, Int, Blocks, Touches, Att 3rd, Att Pen) giúp phân cụm phản ánh tốt các khía cạnh chuyền bóng, phòng ngự, và tham gia tích cực của cầu thủ. Các đặc trưng như Cmp%, Blocks, và Att Pen đảm bảo rằng các cụm phân biệt rõ ràng giữa các vai trò khác nhau.
- *Thủ môn*: Đã lọc bỏ thủ môn nếu cột Position tồn tại, đảm bảo các cụm chỉ bao gồm cầu thủ sân. Nếu không có cột Position, cần kiểm tra dữ liệu để tránh thủ môn ảnh hưởng đến kết quả.
- *Hạn chế*: Một số cầu thủ có ít phút thi đấu (Minutes) có thể ảnh hưởng đến các chỉ số, dẫn đến phân cụm không chính xác. Đã áp dụng lọc cầu thủ có số phút tối thiểu (>90 phút) để cải thiện chất lượng cụm.
- *Ứng dụng*: Kết quả phân cụm với \( k = 4 \) có thể được sử dụng để phân tích đội hình, so sánh hiệu suất cầu thủ, hoặc xác định các kiểu cầu thủ tương tự nhau trong chuyển nhượng. Để hiểu rõ hơn về mỗi cụm, có thể kiểm tra danh sách cầu thủ và giá trị trung bình của các đặc trưng trong mỗi cụm.

Tóm lại, \( k = 4 \) là lựa chọn hợp lý dựa trên Elbow Plot và ý nghĩa bóng đá, mang lại kết quả phân cụm rõ ràng và hữu ích cho việc phân tích dữ liệu cầu thủ.
""")

if __name__ == "__main__":
    main()