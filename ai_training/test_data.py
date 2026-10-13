import pandas as pd

df = pd.read_json("scam-dialogues-train-text.json")

print("=== Số dòng, số cột ===")
print(df.shape)

print("\n=== Phân bố nhãn (có nhãn 0 không?) ===")
print(df["label"].value_counts())

print("\n=== Các loại lừa đảo ===")
print(df["type"].value_counts())

print("\n=== Độ dài hội thoại (số ký tự) ===")
print(df["dialogue"].str.len().describe())

print("\n=== Số hội thoại trùng y hệt ===")
print(df["dialogue"].duplicated().sum())

print("\n=== Đọc thử 1 hội thoại ===")
print(df.loc[0, "dialogue"])