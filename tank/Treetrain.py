import os
import glob
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = r"C:\Users\kai\Desktop\Machinelearning\TANK\tree\treedata"
OUT_DIR = DATA_DIR

LABEL_GOAL = "action_goal"   # 1 ammo / 2 fuel / 3 dodge
LABEL_MOVE = "action_move"
LABEL_FIRE = "action_fire"

WEIGHT_COL = "sample_weight"  # 沒有就補 1.0

CANDIDATE_FEATURES = [
    "team_id",

    "self_x", "self_y", "self_vx", "self_vy",
    "self_life", "self_fuel", "self_ammo",
    "self_fuel_ratio", "self_ammo_ratio",

    "enemy_x", "enemy_y", "enemy_vx", "enemy_vy",
    "dx", "dy", "dist",

    "turret_angle", "angle_to_enemy", "aim_error_deg",
    "los_ok",

    "nearest_fuel_dist", "nearest_fuel_x", "nearest_fuel_y",
    "nearest_ammo_dist", "nearest_ammo_x", "nearest_ammo_y",
    "is_near_supply",

    "fuel_dx", "fuel_dy",
    "ammo_dx", "ammo_dy",

    "enemy_bullet_count",
    "nearest_bullet_dist", "nearest_bullet_angle",
    "danger_min_dist", "danger_frames_to_closest",

    "can_fire",
]

DIST_COLS_WITH_MINUS1 = [
    "nearest_fuel_dist",
    "nearest_ammo_dist",
    "nearest_bullet_dist",
    "danger_min_dist",
]

# ===================== 讀檔 =====================
def load_all_csv(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"找不到任何 CSV：{data_dir}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8")
        df["__source_file__"] = os.path.basename(p)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def add_team_id(df: pd.DataFrame) -> pd.DataFrame:
    if "team_id" not in df.columns:
        if "team" in df.columns:
            df["team_id"] = df["team"].map({"Green": 0, "Blue": 1}).fillna(-1).astype(int)
        else:
            df["team_id"] = -1
    return df

def ensure_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = 1.0
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0).astype(float)
    df.loc[df[WEIGHT_COL] <= 0, WEIGHT_COL] = 1.0
    return df

def clean_and_prepare(df: pd.DataFrame):
    needed = [LABEL_GOAL, LABEL_MOVE, LABEL_FIRE]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少 label 欄位：{miss}，請確認蒐集程式有輸出這些欄位。")

    df = add_team_id(df)
    df = ensure_sample_weight(df)

    # -1 距離 -> 99999 + has_flag（避免 -1 被樹當成超近）
    for col in DIST_COLS_WITH_MINUS1:
        if col in df.columns:
            flag = f"has_{col}"
            x = pd.to_numeric(df[col], errors="coerce")
            df[flag] = (x >= 0).astype(int)
            df[col] = x.replace(-1, 99999)

    # features
    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    for col in DIST_COLS_WITH_MINUS1:
        flag = f"has_{col}"
        if flag in df.columns and flag not in feature_cols:
            feature_cols.append(flag)

    if not feature_cols:
        raise ValueError("找不到任何可用特徵欄位，請確認 CSV 欄位是否正確。")

    # 全部轉數值
    for c in feature_cols + [LABEL_GOAL, LABEL_MOVE, LABEL_FIRE, WEIGHT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=feature_cols + [LABEL_GOAL, LABEL_MOVE, LABEL_FIRE, WEIGHT_COL]).reset_index(drop=True)

    # label int
    df[LABEL_GOAL] = df[LABEL_GOAL].astype(int)
    df[LABEL_MOVE] = df[LABEL_MOVE].astype(int)
    df[LABEL_FIRE] = df[LABEL_FIRE].astype(int)

    # 防呆：goal 範圍
    df = df[df[LABEL_GOAL].isin([0, 1, 2, 3])].reset_index(drop=True)

    return df, feature_cols

# ===================== 訓練核心 =====================
def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    model_name: str,
    out_dir: str,
    class_weight=None,
    max_depth=14,
    min_leaf=30,
    min_samples_split=2,
):
    strat = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=42,
        stratify=strat
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight=class_weight
    )

    clf.fit(X_train, y_train, sample_weight=w_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred, sample_weight=w_test)

    print("\n" + "=" * 80)
    print(f"[{model_name}] Weighted Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop Feature Importances:")
    print(importances.head(20))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, model_name)
    joblib.dump({"model": clf, "feature_cols": list(X.columns)}, out_path)
    print(f"\nSaved: {out_path}")
    return clf

# ===================== 主流程 =====================
def main():
    print(f"[INFO] Loading CSVs from: {DATA_DIR}")
    df = load_all_csv(DATA_DIR)
    print(f"[INFO] Loaded rows: {len(df)}")

    df, feature_cols = clean_and_prepare(df)
    print(f"[INFO] After clean rows: {len(df)}")
    print(f"[INFO] Features used ({len(feature_cols)}): {feature_cols}")

    # 分佈稽核
    print("\n[INFO] action_goal distribution (raw):")
    print(df[LABEL_GOAL].value_counts(normalize=True).sort_index())

    # dodge 資料量檢查（很重要）
    dodge_cnt = int((df[LABEL_GOAL] == 3).sum())
    print(f"\n[INFO] dodge(goal=3) rows: {dodge_cnt}")
    if dodge_cnt < 1500:
        print("[WARN] goal=3(dodge) 樣本偏少，之後推論很容易不會閃子彈。建議多跑幾場蒐集。")

    print("\n[INFO] action_fire=1 rate (raw):")
    print(float(df[LABEL_FIRE].mean()))

    if "can_fire" in df.columns:
        can_fire_rate = float(df["can_fire"].mean())
        print("\n[INFO] can_fire==1 rate:")
        print(can_fire_rate)

        df_can = df[df["can_fire"] == 1].copy()
        if len(df_can) > 0:
            print("\n[INFO] action_fire=1 rate (can_fire==1 subset):")
            print(float(df_can[LABEL_FIRE].mean()))

    X = df[feature_cols].copy()
    w = df[WEIGHT_COL].copy()

    # ===================== 1) Goal model（4 類：fight/ammo/fuel/dodge） =====================
    y_goal = df[LABEL_GOAL]
    train_classifier(
        X, y_goal, w,
        "goal_model.joblib", OUT_DIR,
        class_weight=None,
        max_depth=10,
        min_leaf=80
    )

    # ===================== 2) Move models（依 goal 分流） =====================
    # (A) dodge move
    df_dodge = df[df[LABEL_GOAL] == 3].copy()
    if len(df_dodge) >= 1500 and df_dodge[LABEL_MOVE].nunique() > 1:
        print(f"\n[INFO] Dodge subset rows: {len(df_dodge)}")
        # dodge 很關鍵：通常希望更敏感，所以 leaf 小一點
        train_classifier(
            df_dodge[feature_cols],
            df_dodge[LABEL_MOVE],
            df_dodge[WEIGHT_COL],
            "move_dodge_model.joblib", OUT_DIR,
            class_weight=None,
            max_depth=12,
            min_leaf=25
        )
    else:
        print("\n[WARN] dodge 子集合資料太少或類別太少，略過 move_dodge_model。")

    # (B) supply move (goal=1 or 2)
    df_sup = df[df[LABEL_GOAL].isin([1, 2])].copy()
    if len(df_sup) >= 1500 and df_sup[LABEL_MOVE].nunique() > 1:
        print(f"\n[INFO] Supply subset rows: {len(df_sup)}")

        # 讓「靠近補包」樣本更重要，move 更黏補包
        w_sup = df_sup[WEIGHT_COL].copy()
        if "is_near_supply" in df_sup.columns:
            w_sup = w_sup * (1.0 + 0.8 * df_sup["is_near_supply"].astype(float))

        train_classifier(
            df_sup[feature_cols],
            df_sup[LABEL_MOVE],
            w_sup,
            "move_supply_model.joblib", OUT_DIR,
            class_weight=None,
            max_depth=10,
            min_leaf=30
        )
    else:
        print("\n[WARN] supply 子集合資料太少或類別太少，略過 move_supply_model。")

    # ===================== 3) Fire model（只用 can_fire==1） =====================
    if "can_fire" in df.columns:
        df_fire = df[df["can_fire"] == 1].copy()
    else:
        df_fire = df.copy()

    if len(df_fire) < 1500 or df_fire[LABEL_FIRE].nunique() <= 1:
        print("\n[WARN] fire 訓練資料太少或類別太少，略過 fire_model。")
    else:
        print(f"\n[INFO] Fire training subset rows: {len(df_fire)}")
        print(f"[INFO] action_fire=1 rate (train subset): {float(df_fire[LABEL_FIRE].mean()):.4f}")

        # 你需求是「盡量一直射」：fire=1 權重加大
        w_fire = df_fire[WEIGHT_COL].copy()
        w_fire = w_fire * (1.0 + 2.0 * df_fire[LABEL_FIRE].astype(float))  # fire=1 變 3 倍權重

        train_classifier(
            df_fire[feature_cols],
            df_fire[LABEL_FIRE],
            w_fire,
            "fire_model.joblib", OUT_DIR,
            class_weight=None,
            max_depth=8,
            min_leaf=60
        )

    print("\n[DONE] Training finished.")
    print("推論端建議：goal_model -> (goal=3用move_dodge / goal=1,2用move_supply ) -> fire_model（can_fire==1 才用）")

if __name__ == "__main__":
    main()
