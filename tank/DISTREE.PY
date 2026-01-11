import os
import glob
import joblib
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree


MODEL_DIR = r"C:\Users\kai\Desktop\Machinelearning\TANK\tree\treedata"
OUT_DIR   = os.path.join(MODEL_DIR, "tree_plots")


REPORT_MAX_DEPTH = 4


BASE_FIGSIZE = (28, 14)  
DPI = 200                 


def load_model_bundle(path: str):

    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        clf = obj["model"]
        feature_cols = obj.get("feature_cols", None)
    else:
        clf = obj
        feature_cols = None
    return clf, feature_cols


def plot_and_save_tree(clf, feature_names, class_names, out_base: str, max_depth=None):

    title = os.path.basename(out_base)
    if max_depth is not None:
        title += f" (top {max_depth} levels)"

    plt.figure(figsize=BASE_FIGSIZE)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
        max_depth=max_depth,
    )
    plt.title(title)
    plt.tight_layout()

    png_path = out_base + (f"_top{max_depth}.png" if max_depth is not None else ".png")
    pdf_path = out_base + (f"_top{max_depth}.pdf" if max_depth is not None else ".pdf")

    plt.savefig(png_path, dpi=DPI)
    plt.savefig(pdf_path)
    plt.close()

    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def infer_class_names(model_filename: str, clf):
    """
    根據檔名推測 class_names；推不到就用 clf.classes_。
    """
    name = os.path.basename(model_filename).lower()

    if "goal" in name:
        # 0 fight / 1 ammo / 2 fuel / 3 dodge
        return ["fight(0)", "ammo(1)", "fuel(2)", "dodge(3)"]

    if "fire" in name:
        return ["no_fire(0)", "fire(1)"]

    # move 的類別通常是 0..K-1，這裡就直接用 clf.classes_
    try:
        return [str(c) for c in clf.classes_]
    except Exception:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib")))
    if not joblib_paths:
        raise FileNotFoundError(f"找不到任何 .joblib：{MODEL_DIR}")

    print(f"[INFO] Found {len(joblib_paths)} joblib models in: {MODEL_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")

    for p in joblib_paths:
        print("\n" + "=" * 80)
        print(f"[INFO] Loading: {p}")

        clf, feature_cols = load_model_bundle(p)

        # feature_names：優先用你存的 feature_cols，沒有就用 f0,f1,...
        if feature_cols is None:
            try:
                n_feat = getattr(clf, "n_features_in_", None)
                if n_feat is None:
                    feature_cols = None
                else:
                    feature_cols = [f"f{i}" for i in range(n_feat)]
            except Exception:
                feature_cols = None

        # class_names：依檔名推測，推不到就用 clf.classes_
        class_names = infer_class_names(p, clf)

        base = os.path.splitext(os.path.basename(p))[0]  # e.g. goal_model
        out_base = os.path.join(OUT_DIR, f"tree_{base}")

        # 1) 整棵樹
        plot_and_save_tree(
            clf,
            feature_names=feature_cols,
            class_names=class_names,
            out_base=out_base,
            max_depth=None
        )

        plot_and_save_tree(
            clf,
            feature_names=feature_cols,
            class_names=class_names,
            out_base=out_base,
            max_depth=REPORT_MAX_DEPTH
        )

    print("\n[DONE] All trees plotted.")


if __name__ == "__main__":
    main()
