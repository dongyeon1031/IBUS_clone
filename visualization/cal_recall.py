import pandas as pd

# CSV 파일 경로 지정
csv_path = "./outputs/results_out.csv"

# CSV 불러오기
df = pd.read_csv(csv_path)

# 문자열을 정수형으로 변환 (예: "5;3;6;8;7" → [5,3,6,8,7])
def parse_top5(s):
    return [int(x) for x in str(s).split(";") if x.isdigit()]

# 각 행별 Top5 리스트 추가
df["Top5_List"] = df["Top5 IDs"].apply(parse_top5)

# ✅ R@1 계산 (True ID == Predict ID)
r1 = (df["True ID"].astype(int) == df["Predict ID"].astype(int)).mean()

# ✅ R@5 계산 (True ID ∈ Top5 IDs)
r5 = df.apply(lambda row: int(row["True ID"]) in row["Top5_List"], axis=1).mean()

print(f"🔹 R@1  = {r1 * 100:.2f}%")
print(f"🔹 R@5  = {r5 * 100:.2f}%")
print(f"Samples = {len(df)}")