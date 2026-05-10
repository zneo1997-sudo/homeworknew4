# -*- coding: utf-8 -*-
"""
Vibe Coding 升级版：图像分类与优化算法可视化平台
适配 GitHub + Streamlit Cloud 部署

项目支持：
1. Least Squares Linear Regression
2. KNN 图像分类：支持示例数据 / 本地数据 / 上传 zip 数据集 / 上传单张图片预测
3. 线性分类器：Softmax / SVM Loss，SGD / Momentum，模板图像可视化
4. SGD 与 Momentum 梯度下降路径对比
5. 不同 Loss 的计算过程演示

推荐项目结构：
project/
├── app.py
├── requirements.txt
└── data/
    └── class4/
        ├── Knife/
        │   ├── 001.jpg
        │   └── ...
        └── Pistol/
            ├── 001.jpg
            └── ...

requirements.txt 建议：
streamlit
numpy
pandas
matplotlib
Pillow
"""

import io
import os
import math
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw, ImageOps


# ============================================================
# 页面配置与全局样式
# ============================================================
st.set_page_config(
    page_title="Vibe Coding 图像分类实验平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6f8fc 0%, #ffffff 100%);
    }
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .card {
        padding: 1.15rem 1.25rem;
        border-radius: 1rem;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.9rem;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .metric-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #111827;
    }
    .metric-label {
        font-size: 0.88rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 通用工具函数
# ============================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LOCAL_DATA_DIRS = [
    Path("data/class4"),
    Path("data/weapon_dataset"),
    Path("class4"),
]


@st.cache_data(show_spinner=False)
def list_local_data_dirs():
    found = []
    for p in DEFAULT_LOCAL_DATA_DIRS:
        if p.exists() and p.is_dir():
            found.append(str(p))
    return found


def normalize_image_array(arr):
    arr = arr.astype(np.float32)
    return arr / 255.0


def display_class_name(name):
    mapping = {
        "Knife": "刀 Knife",
        "knife": "刀 Knife",
        "Pistol": "枪 Pistol",
        "pistol": "枪 Pistol",
        "Gun": "枪 Gun",
        "gun": "枪 Gun",
    }
    return mapping.get(str(name), str(name))


def image_to_array(img, image_size=(32, 32), color_mode="RGB"):
    """把 PIL 图片转换成统一尺寸和颜色模式。"""
    if color_mode == "L":
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32)
    if color_mode == "L":
        arr = arr[..., None]
    return arr


def load_images_from_folder(root_dir, image_size=(32, 32), color_mode="RGB", max_per_class=None):
    """读取按类别分文件夹的数据集。"""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"数据集路径不存在：{root_dir}")

    class_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        # 兼容 zip 解压后多一层根目录的情况
        sub_dirs = [p for p in root_dir.rglob("*") if p.is_dir()]
        class_dirs = [p for p in sub_dirs if any(x.suffix.lower() in IMAGE_EXTS for x in p.iterdir() if x.is_file())]

    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())
    X, y, file_paths = [], [], []
    class_names = []

    for label_id, class_dir in enumerate(class_dirs):
        image_files = [p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if len(image_files) == 0:
            continue
        class_names.append(class_dir.name)
        if max_per_class is not None:
            image_files = image_files[:max_per_class]
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                arr = image_to_array(img, image_size=image_size, color_mode=color_mode)
                X.append(arr)
                y.append(len(class_names) - 1)
                file_paths.append(str(img_path))
            except Exception:
                pass

    if len(X) == 0:
        raise ValueError("没有读取到有效图片。请确认每个类别文件夹中包含 jpg/png/jpeg/bmp/webp 图片。")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), class_names, file_paths


def load_images_from_zip(uploaded_zip, image_size=(32, 32), color_mode="RGB", max_per_class=None):
    """读取上传的图片文件夹 zip。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "dataset.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(Path(tmpdir) / "unzipped")
        root = Path(tmpdir) / "unzipped"

        # 如果压缩包只有一个根目录，就进入该目录
        children = [p for p in root.iterdir() if p.is_dir()]
        if len(children) == 1 and not any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in root.iterdir()):
            root = children[0]
        return load_images_from_folder(root, image_size=image_size, color_mode=color_mode, max_per_class=max_per_class)


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    test_n = max(1, int(len(X) * test_ratio))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    if len(train_idx) == 0:
        train_idx = test_idx
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def flatten_features(X):
    return X.reshape(len(X), -1).astype(np.float32)


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def make_image_grid(images, titles=None, cols=5, cmap=None, figsize_per_cell=2.2):
    n = len(images)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap=cmap or "gray")
        else:
            img_show = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.5 else np.clip(img, 0, 1)
            ax.imshow(img_show, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# 示例数据：自动生成 Knife / Pistol 简笔图，保证网站无外部数据也能运行
# ============================================================
@st.cache_data(show_spinner=False)
def generate_demo_weapon_dataset(n_per_class=50, image_size=(32, 32), color_mode="RGB", seed=7):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    class_names = ["Knife", "Pistol"]

    for cls in range(2):
        for _ in range(n_per_class):
            bg = int(rng.integers(235, 256))
            img = Image.new("RGB", image_size, (bg, bg, bg))
            draw = ImageDraw.Draw(img)
            jitter_x = int(rng.integers(-2, 3))
            jitter_y = int(rng.integers(-2, 3))
            shade = int(rng.integers(30, 95))
            accent = int(rng.integers(110, 190))

            if cls == 0:  # Knife：刀刃 + 手柄
                y = 15 + jitter_y
                draw.polygon([(3+jitter_x, y), (17+jitter_x, y-4), (18+jitter_x, y+2), (4+jitter_x, y+2)], fill=(shade, shade, shade))
                draw.rounded_rectangle((17+jitter_x, y-1, 29+jitter_x, y+5), radius=2, fill=(accent, accent-30, 80))
                draw.ellipse((18+jitter_x, y, 21+jitter_x, y+3), fill=(40, 40, 40))
                for k in range(3):
                    x = 22 + k * 3 + jitter_x
                    draw.line((x, y, x-1, y+4), fill=(70, 70, 70), width=1)
            else:  # Pistol：枪身 + 握把 + 枪托/枪管
                x0, y0 = 7 + jitter_x, 12 + jitter_y
                draw.rectangle((x0, y0, x0+15, y0+5), fill=(shade, shade, shade))
                draw.rectangle((x0+14, y0+1, x0+25, y0+3), fill=(shade, shade, shade))
                draw.rectangle((x0+6, y0+5, x0+11, y0+15), fill=(shade+20, shade+20, shade+20))
                draw.rectangle((x0+1, y0+5, x0+7, y0+8), fill=(shade, shade, shade))
                draw.line((x0+18, y0+5, x0+24, y0+13), fill=(shade, shade, shade), width=2)
                draw.line((x0+24, y0+13, x0+27, y0+7), fill=(shade, shade, shade), width=2)

            # 轻微旋转和噪声
            angle = float(rng.normal(0, 4))
            img = img.rotate(angle, fillcolor=(bg, bg, bg))
            arr = np.array(img, dtype=np.float32)
            noise = rng.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            if color_mode == "L":
                arr = np.array(Image.fromarray(arr).convert("L"))[..., None]
            images.append(arr.astype(np.float32))
            labels.append(cls)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), class_names


# ============================================================
# 数据加载 UI
# ============================================================
def dataset_loader_ui(image_size=(32, 32), color_mode="RGB", key_prefix="data", max_per_class_default=200):
    st.markdown("### 数据来源")
    options = ["使用网站示例数据"]
    local_dirs = list_local_data_dirs()
    if local_dirs:
        options.append("使用项目内置本地数据")
    options.append("上传图片文件夹 zip")

    source = st.radio("请选择数据来源", options, horizontal=True, key=f"{key_prefix}_source")
    max_per_class = st.slider("每类最多读取图片数", 10, 1000, max_per_class_default, 10, key=f"{key_prefix}_max")

    if source == "使用网站示例数据":
        X, y, class_names = generate_demo_weapon_dataset(
            n_per_class=min(max_per_class, 200),
            image_size=image_size,
            color_mode=color_mode,
        )
        file_paths = ["demo"] * len(X)
        st.success("已加载网站示例数据。没有真实数据时也可以完整体验流程。")
        return X, y, class_names, file_paths

    if source == "使用项目内置本地数据":
        selected_dir = st.selectbox("选择项目内的数据目录", local_dirs, key=f"{key_prefix}_local_dir")
        X, y, class_names, file_paths = load_images_from_folder(
            selected_dir,
            image_size=image_size,
            color_mode=color_mode,
            max_per_class=max_per_class,
        )
        st.success(f"已加载本地数据目录：{selected_dir}")
        return X, y, class_names, file_paths

    uploaded_zip = st.file_uploader(
        "上传图片数据集 zip。要求：每个类别一个文件夹，例如 Knife/ 与 Pistol/。",
        type=["zip"],
        key=f"{key_prefix}_zip",
    )
    if uploaded_zip is None:
        st.info("请先上传 zip 数据集。")
        st.stop()
    X, y, class_names, file_paths = load_images_from_zip(
        uploaded_zip,
        image_size=image_size,
        color_mode=color_mode,
        max_per_class=max_per_class,
    )
    st.success("上传数据集读取成功。")
    return X, y, class_names, file_paths


def show_dataset_summary(X, y, class_names):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("图片数量", len(X))
    c2.metric("类别数量", len(class_names))
    c3.metric("图像尺寸", f"{X.shape[1]}×{X.shape[2]}")
    c4.metric("通道数", X.shape[3])

    dist = pd.DataFrame({
        "类别": [display_class_name(class_names[i]) for i in range(len(class_names))],
        "数量": [int(np.sum(y == i)) for i in range(len(class_names))],
    })
    st.dataframe(dist, use_container_width=True, hide_index=True)

    sample_images, sample_titles = [], []
    for i, name in enumerate(class_names):
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            sample_images.append(X[idx[0]])
            sample_titles.append(display_class_name(name))
    st.pyplot(make_image_grid(sample_images, sample_titles, cols=len(sample_images)))


# ============================================================
# 1. 最小二乘线性回归
# ============================================================
def parse_xy_text(x_text, y_text):
    x = np.array([float(v.strip()) for v in x_text.split(",") if v.strip() != ""], dtype=np.float64)
    y = np.array([float(v.strip()) for v in y_text.split(",") if v.strip() != ""], dtype=np.float64)
    return x, y


def least_squares_fit(x, y):
    X = np.c_[np.ones(len(x)), x]
    try:
        theta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    b, w = float(theta[0]), float(theta[1])
    return w, b, X, theta


def show_regression():
    st.markdown('<div class="main-title">1. Least Squares Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">支持示例数据、手动输入和 CSV 上传，展示闭式解、拟合线、残差和计算矩阵。</div>', unsafe_allow_html=True)

    data_source = st.radio("选择数据来源", ["交互生成示例数据", "手动输入", "上传 CSV"], horizontal=True)

    if data_source == "交互生成示例数据":
        col1, col2, col3, col4 = st.columns(4)
        n = col1.slider("样本数量", 10, 200, 50)
        noise = col2.slider("噪声强度", 0.0, 10.0, 2.0)
        w_true = col3.slider("真实斜率 w", -10.0, 10.0, 3.0)
        b_true = col4.slider("真实截距 b", -10.0, 10.0, 5.0)
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, n)
        y = w_true * x + b_true + rng.normal(0, noise, n)
    elif data_source == "手动输入":
        x_text = st.text_area("请输入 X，逗号分隔", "1,2,3,4,5,6,7,8")
        y_text = st.text_area("请输入 y，逗号分隔", "2,4,5,4,6,8,9,10")
        try:
            x, y = parse_xy_text(x_text, y_text)
        except Exception as e:
            st.error(f"输入解析失败：{e}")
            st.stop()
    else:
        uploaded_csv = st.file_uploader("上传 CSV，要求至少包含 x 和 y 两列", type=["csv"])
        if uploaded_csv is None:
            st.info("请先上传 CSV。")
            st.stop()
        df = pd.read_csv(uploaded_csv)
        if "x" not in df.columns or "y" not in df.columns:
            st.error("CSV 必须包含 x 和 y 两列。")
            st.stop()
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)

    if len(x) != len(y) or len(x) < 2:
        st.error("X 和 y 长度必须相同，且至少包含两个样本。")
        st.stop()

    w, b, X_mat, theta = least_squares_fit(x, y)
    y_pred = w * x + b
    mse = float(np.mean((y - y_pred) ** 2))

    c1, c2, c3 = st.columns(3)
    c1.metric("斜率 w", f"{w:.6f}")
    c2.metric("截距 b", f"{b:.6f}")
    c3.metric("MSE", f"{mse:.6f}")

    tab1, tab2, tab3 = st.tabs(["拟合可视化", "残差分析", "计算过程"])
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(x, y, label="真实数据")
        x_line = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(x_line, w * x_line + b, label="最小二乘拟合线")
        ax.set_title(f"y = {w:.4f}x + {b:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        residual = y - y_pred
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.bar(np.arange(len(residual)), residual)
        ax.set_title("Residual = y_true - y_pred")
        ax.set_xlabel("样本编号")
        ax.set_ylabel("残差")
        st.pyplot(fig)
        st.dataframe(pd.DataFrame({"x": x, "y_true": y, "y_pred": y_pred, "residual": residual}), use_container_width=True)

    with tab3:
        st.latex(r"\theta = (X^T X)^{-1}X^T y")
        st.write("设计矩阵 X：")
        st.dataframe(pd.DataFrame(X_mat, columns=["1", "x"]).head(20), use_container_width=True)
        st.write("XᵀX：")
        st.dataframe(pd.DataFrame(X_mat.T @ X_mat), use_container_width=True)
        st.write("参数 θ = [b, w]：")
        st.dataframe(pd.DataFrame(theta.reshape(1, -1), columns=["b", "w"]), use_container_width=True)


# ============================================================
# 2. KNN 图像分类
# ============================================================
def distance_matrix(X_train, x, metric="euclidean"):
    if metric == "euclidean":
        return np.sqrt(np.sum((X_train - x) ** 2, axis=1))
    if metric == "manhattan":
        return np.sum(np.abs(X_train - x), axis=1)
    if metric == "cosine":
        denom = (np.linalg.norm(X_train, axis=1) * (np.linalg.norm(x) + 1e-8)) + 1e-8
        return 1.0 - (X_train @ x) / denom
    raise ValueError("未知距离度量")


def knn_predict_single(X_train, y_train, x, k=3, metric="euclidean"):
    dist = distance_matrix(X_train, x, metric=metric)
    nn_idx = np.argsort(dist)[:k]
    labels = y_train[nn_idx]
    values, counts = np.unique(labels, return_counts=True)
    pred = int(values[np.argmax(counts)])
    return pred, nn_idx, dist[nn_idx]


def knn_predict_batch(X_train, y_train, X_test, k=3, metric="euclidean"):
    preds = []
    for x in X_test:
        pred, _, _ = knn_predict_single(X_train, y_train, x, k=k, metric=metric)
        preds.append(pred)
    return np.array(preds, dtype=np.int64)


def show_knn():
    st.markdown('<div class="main-title">2. KNN 图像分类</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">支持 Knife / Pistol 文件夹数据集、不同 K 值对比、最近邻可视化和用户上传图片预测。</div>', unsafe_allow_html=True)

    X_img, y, class_names, file_paths = dataset_loader_ui(image_size=(32, 32), color_mode="L", key_prefix="knn", max_per_class_default=200)
    show_dataset_summary(X_img, y, class_names)

    X = flatten_features(normalize_image_array(X_img))
    test_ratio = st.slider("测试集比例", 0.1, 0.5, 0.25, 0.05, key="knn_test_ratio")
    seed = st.number_input("随机种子", 0, 9999, 42, 1, key="knn_seed")
    metric = st.selectbox("距离度量", ["euclidean", "manhattan", "cosine"], key="knn_metric")

    X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, test_ratio=test_ratio, seed=int(seed))
    X_train = flatten_features(normalize_image_array(X_train_img))
    X_test = flatten_features(normalize_image_array(X_test_img))

    tab1, tab2, tab3 = st.tabs(["不同 K 对比", "单个 K 预测样例", "上传图片预测"])

    with tab1:
        k_text = st.text_input("输入多个 K 值，用逗号分隔", "1,3,5,7,9")
        k_values = [int(v.strip()) for v in k_text.split(",") if v.strip().isdigit() and int(v.strip()) > 0]
        if st.button("运行 KNN 对比", type="primary"):
            rows = []
            for k in k_values:
                y_pred = knn_predict_batch(X_train, y_train, X_test, k=k, metric=metric)
                acc = float(np.mean(y_pred == y_test))
                rows.append({"K": k, "Accuracy": acc})
            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(result_df["K"], result_df["Accuracy"], marker="o")
            ax.set_title("不同 K 值的准确率对比")
            ax.set_xlabel("K")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    with tab2:
        k = st.slider("选择 K", 1, 21, 3, 2, key="knn_single_k")
        if len(X_test) > 0:
            sample_idx = st.slider("选择测试样本编号", 0, len(X_test) - 1, 0)
            pred, nn_idx, nn_dist = knn_predict_single(X_train, y_train, X_test[sample_idx], k=k, metric=metric)
            st.success(f"预测类别：{display_class_name(class_names[pred])}；真实类别：{display_class_name(class_names[y_test[sample_idx]])}")
            imgs = [X_test_img[sample_idx]] + [X_train_img[i] for i in nn_idx]
            titles = ["输入图像"] + [f"邻居{i+1}\n{display_class_name(class_names[y_train[idx]])}\nd={nn_dist[i]:.3f}" for i, idx in enumerate(nn_idx)]
            st.pyplot(make_image_grid(imgs, titles=titles, cols=min(k + 1, 6), cmap="gray"))

    with tab3:
        uploaded_img = st.file_uploader("上传一张 jpg/png 图片进行预测", type=["jpg", "jpeg", "png", "bmp", "webp"], key="knn_predict_img")
        k_upload = st.slider("上传图片预测 K", 1, 21, 3, 2, key="knn_upload_k")
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            arr = image_to_array(img, image_size=(32, 32), color_mode="L")
            vec = flatten_features(normalize_image_array(arr[None, ...]))[0]
            pred, nn_idx, nn_dist = knn_predict_single(X_train, y_train, vec, k=k_upload, metric=metric)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(img, caption="上传图片", use_container_width=True)
                st.success(f"预测结果：{display_class_name(class_names[pred])}")
            with c2:
                imgs = [arr] + [X_train_img[i] for i in nn_idx]
                titles = ["处理后输入"] + [f"{display_class_name(class_names[y_train[idx]])}\nd={nn_dist[i]:.3f}" for i, idx in enumerate(nn_idx)]
                st.pyplot(make_image_grid(imgs, titles=titles, cols=min(k_upload + 1, 5), cmap="gray"))


# ============================================================
# 3. 线性分类器：Softmax / SVM + SGD / Momentum
# ============================================================
def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax_loss_and_grad(W, b, X, y, reg=1e-4):
    N = X.shape[0]
    scores = X @ W + b
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12)) + 0.5 * reg * np.sum(W * W)
    ds = probs
    ds[np.arange(N), y] -= 1
    ds /= N
    dW = X.T @ ds + reg * W
    db = ds.sum(axis=0)
    return float(loss), dW, db


def svm_loss_and_grad(W, b, X, y, reg=1e-4, delta=1.0):
    N = X.shape[0]
    scores = X @ W + b
    correct = scores[np.arange(N), y][:, None]
    margins = np.maximum(0, scores - correct + delta)
    margins[np.arange(N), y] = 0
    loss = np.mean(np.sum(margins, axis=1)) + 0.5 * reg * np.sum(W * W)

    binary = (margins > 0).astype(np.float32)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(N), y] = -row_sum
    binary /= N
    dW = X.T @ binary + reg * W
    db = binary.sum(axis=0)
    return float(loss), dW, db


def train_linear_classifier(X_train, y_train, X_test, y_test, num_classes, loss_type="softmax", optimizer="momentum", lr=0.05, epochs=50, batch_size=32, reg=1e-4, momentum=0.9, seed=42):
    rng = np.random.default_rng(seed)
    D = X_train.shape[1]
    W = rng.normal(0, 0.01, size=(D, num_classes)).astype(np.float32)
    b = np.zeros(num_classes, dtype=np.float32)
    vW = np.zeros_like(W)
    vb = np.zeros_like(b)
    history = []

    loss_fn = softmax_loss_and_grad if loss_type == "softmax" else svm_loss_and_grad
    N = len(X_train)

    for epoch in range(epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            loss, dW, db = loss_fn(W, b, xb, yb, reg=reg)
            if optimizer == "momentum":
                vW = momentum * vW - lr * dW
                vb = momentum * vb - lr * db
                W += vW
                b += vb
            else:
                W -= lr * dW
                b -= lr * db

        train_scores = X_train @ W + b
        test_scores = X_test @ W + b
        train_acc = float(np.mean(np.argmax(train_scores, axis=1) == y_train))
        test_acc = float(np.mean(np.argmax(test_scores, axis=1) == y_test))
        full_loss, _, _ = loss_fn(W, b, X_train, y_train, reg=reg)
        history.append({"epoch": epoch + 1, "loss": full_loss, "train_acc": train_acc, "test_acc": test_acc})

    return W, b, pd.DataFrame(history)


def visualize_templates(W, image_shape, class_names):
    H, Wimg, C = image_shape
    templates = []
    for c in range(len(class_names)):
        img = W[:, c].reshape(H, Wimg, C)
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        templates.append(img)
    titles = [display_class_name(n) for n in class_names]
    return make_image_grid(templates, titles=titles, cols=min(len(class_names), 5))


def show_linear_classifier():
    st.markdown('<div class="main-title">3. CIFAR 风格线性分类器与模板图像可视化</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">这里使用通用图片文件夹数据集。Knife/Pistol 会作为二分类；也可以上传更多类别。</div>', unsafe_allow_html=True)

    X_img, y, class_names, _ = dataset_loader_ui(image_size=(32, 32), color_mode="RGB", key_prefix="linear", max_per_class_default=200)
    show_dataset_summary(X_img, y, class_names)

    if len(class_names) < 2:
        st.error("线性分类器至少需要 2 个类别。")
        st.stop()

    test_ratio = st.slider("测试集比例", 0.1, 0.5, 0.25, 0.05, key="linear_test")
    seed = st.number_input("随机种子", 0, 9999, 42, 1, key="linear_seed")
    X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, test_ratio=test_ratio, seed=int(seed))

    X_train = flatten_features(normalize_image_array(X_train_img))
    X_test = flatten_features(normalize_image_array(X_test_img))
    X_train, X_test, mean, std = standardize_train_test(X_train, X_test)

    st.markdown("### 训练参数")
    c1, c2, c3, c4 = st.columns(4)
    loss_type = c1.selectbox("Loss 类型", ["softmax", "svm"], index=0)
    optimizer = c2.selectbox("优化器", ["sgd", "momentum"], index=1)
    lr = c3.number_input("学习率", 0.0001, 2.0, 0.05, 0.01, format="%.4f")
    epochs = c4.slider("训练轮数", 5, 300, 60, 5)
    c5, c6, c7 = st.columns(3)
    batch_size = c5.slider("Batch Size", 4, 256, 32, 4)
    reg = c6.number_input("L2 正则强度", 0.0, 1.0, 0.0001, 0.0001, format="%.5f")
    momentum_value = c7.slider("Momentum 系数", 0.0, 0.99, 0.9, 0.01)

    if st.button("开始训练线性分类器", type="primary"):
        with st.spinner("正在训练线性分类器..."):
            W, b, hist = train_linear_classifier(
                X_train, y_train, X_test, y_test,
                num_classes=len(class_names),
                loss_type=loss_type,
                optimizer=optimizer,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                reg=reg,
                momentum=momentum_value,
                seed=int(seed),
            )
        st.session_state["linear_model"] = {"W": W, "b": b, "hist": hist, "mean": mean, "std": std, "class_names": class_names}
        st.success("训练完成。")

    if "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
        W, b, hist = model["W"], model["b"], model["hist"]
        c1, c2, c3 = st.columns(3)
        c1.metric("最终 Loss", f"{hist['loss'].iloc[-1]:.4f}")
        c2.metric("训练准确率", f"{hist['train_acc'].iloc[-1]*100:.2f}%")
        c3.metric("测试准确率", f"{hist['test_acc'].iloc[-1]*100:.2f}%")

        tab1, tab2, tab3 = st.tabs(["训练曲线", "模板图像", "上传图片预测"])
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(hist["epoch"], hist["loss"], label="loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("训练 Loss 曲线")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(hist["epoch"], hist["train_acc"], label="train_acc")
            ax.plot(hist["epoch"], hist["test_acc"], label="test_acc")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("准确率曲线")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.caption("线性分类器的权重 W 可以 reshape 回图片尺寸，近似表示每个类别学到的模板。")
            st.pyplot(visualize_templates(W, image_shape=(32, 32, 3), class_names=class_names))

        with tab3:
            uploaded_img = st.file_uploader("上传一张图片，让线性分类器预测", type=["jpg", "jpeg", "png", "bmp", "webp"], key="linear_predict")
            if uploaded_img is not None:
                img = Image.open(uploaded_img)
                arr = image_to_array(img, image_size=(32, 32), color_mode="RGB")
                vec = flatten_features(normalize_image_array(arr[None, ...]))
                vec = (vec - model["mean"]) / model["std"]
                scores = vec @ W + b
                pred = int(np.argmax(scores[0]))
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img, caption="上传图片", use_container_width=True)
                    st.success(f"预测结果：{display_class_name(class_names[pred])}")
                with c2:
                    score_df = pd.DataFrame({
                        "类别": [display_class_name(n) for n in class_names],
                        "score": scores[0],
                    }).sort_values("score", ascending=False)
                    st.dataframe(score_df, use_container_width=True, hide_index=True)


# ============================================================
# 4. SGD / Momentum 梯度下降可视化
# ============================================================
def f_2d(x, y):
    return x ** 2 + 5 * y ** 2


def grad_2d(x, y):
    return np.array([2 * x, 10 * y], dtype=np.float64)


def run_gd_path(x0, y0, lr, steps, optimizer="sgd", momentum=0.9):
    pos = np.array([x0, y0], dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    path = []
    for i in range(steps):
        g = grad_2d(pos[0], pos[1])
        if optimizer == "momentum":
            v = momentum * v - lr * g
            pos = pos + v
        else:
            pos = pos - lr * g
        path.append({"step": i + 1, "x": pos[0], "y": pos[1], "loss": f_2d(pos[0], pos[1])})
    return pd.DataFrame(path)


def show_gradient_descent():
    st.markdown('<div class="main-title">4. SGD / Momentum 梯度下降可视化</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">在二维函数 f(x,y)=x²+5y² 上对比普通 SGD 与 Momentum 的下降路径。</div>', unsafe_allow_html=True)
    st.latex(r"f(x,y)=x^2+5y^2")

    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.slider("初始 x", -5.0, 5.0, 4.0, 0.1)
    y0 = c2.slider("初始 y", -5.0, 5.0, 4.0, 0.1)
    lr = c3.slider("学习率", 0.001, 0.3, 0.08, 0.001)
    steps = c4.slider("迭代步数", 5, 200, 50, 5)
    momentum = st.slider("Momentum 系数", 0.0, 0.99, 0.9, 0.01, key="gd_momentum")

    sgd_path = run_gd_path(x0, y0, lr, steps, optimizer="sgd", momentum=momentum)
    mom_path = run_gd_path(x0, y0, lr, steps, optimizer="momentum", momentum=momentum)

    tab1, tab2 = st.tabs(["下降路径", "Loss 曲线"])
    with tab1:
        grid_x = np.linspace(-5, 5, 120)
        grid_y = np.linspace(-5, 5, 120)
        XX, YY = np.meshgrid(grid_x, grid_y)
        ZZ = f_2d(XX, YY)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.contour(XX, YY, ZZ, levels=30, alpha=0.7)
        ax.plot(sgd_path["x"], sgd_path["y"], marker="o", markersize=3, label="SGD")
        ax.plot(mom_path["x"], mom_path["y"], marker="o", markersize=3, label="Momentum")
        ax.scatter([0], [0], marker="*", s=120, label="最优点")
        ax.set_title("优化路径对比")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(sgd_path["step"], sgd_path["loss"], label="SGD")
        ax.plot(mom_path["step"], mom_path["loss"], label="Momentum")
        ax.set_title("Loss 下降曲线")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)


# ============================================================
# 5. Loss 计算演示
# ============================================================
def show_loss_demo():
    st.markdown('<div class="main-title">5. 不同 Loss 损失计算过程演示</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">通过手动输入预测值或分类分数，展示 MSE、SVM Hinge Loss、Softmax Cross Entropy 的计算过程。</div>', unsafe_allow_html=True)

    loss_type = st.selectbox("选择 Loss", ["MSE Loss", "SVM Hinge Loss", "Softmax Cross Entropy"])

    if loss_type == "MSE Loss":
        y_true_text = st.text_input("y_true，逗号分隔", "3,5,7")
        y_pred_text = st.text_input("y_pred，逗号分隔", "2.5,5.5,8")
        try:
            y_true = np.array([float(v.strip()) for v in y_true_text.split(",")])
            y_pred = np.array([float(v.strip()) for v in y_pred_text.split(",")])
            err = y_true - y_pred
            sq = err ** 2
            loss = float(np.mean(sq))
            st.latex(r"MSE=\frac{1}{N}\sum_i(y_i-\hat y_i)^2")
            st.metric("MSE", f"{loss:.6f}")
            st.dataframe(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "error": err, "squared_error": sq}), use_container_width=True)
        except Exception as e:
            st.error(f"计算失败：{e}")

    elif loss_type == "SVM Hinge Loss":
        scores_text = st.text_input("输入分类分数 scores，逗号分隔", "2.1, 5.2, 1.3")
        true_idx = st.number_input("正确类别索引，从 0 开始", 0, 20, 1, 1)
        delta = st.slider("margin Δ", 0.1, 5.0, 1.0, 0.1)
        try:
            scores = np.array([float(v.strip()) for v in scores_text.split(",")])
            y = int(true_idx)
            correct = scores[y]
            margins = np.maximum(0, scores - correct + delta)
            margins[y] = 0
            loss = float(np.sum(margins))
            st.latex(r"L_i=\sum_{j\ne y_i}\max(0, s_j-s_{y_i}+\Delta)")
            st.metric("SVM Loss", f"{loss:.6f}")
            st.dataframe(pd.DataFrame({"class": np.arange(len(scores)), "score": scores, "margin": margins}), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"计算失败：{e}")

    else:
        scores_text = st.text_input("输入分类分数 scores，逗号分隔", "2.1, 5.2, 1.3")
        true_idx = st.number_input("正确类别索引，从 0 开始", 0, 20, 1, 1, key="softmax_y")
        try:
            scores = np.array([float(v.strip()) for v in scores_text.split(",")])
            y = int(true_idx)
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores)
            loss = float(-np.log(probs[y] + 1e-12))
            st.latex(r"p_j=\frac{e^{s_j}}{\sum_k e^{s_k}},\quad L=-\log(p_y)")
            st.metric("Cross Entropy Loss", f"{loss:.6f}")
            st.dataframe(pd.DataFrame({"class": np.arange(len(scores)), "score": scores, "exp(score-shift)": exp_scores, "probability": probs}), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"计算失败：{e}")


# ============================================================
# 首页与部署说明
# ============================================================
def show_home():
    st.markdown('<div class="main-title">Vibe Coding 图像分类与优化算法可视化平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">面向课程作业的升级版 Streamlit Web App：支持示例数据、Knife/Pistol 图片数据集上传、KNN、线性分类器、模板可视化、优化过程和 Loss 计算演示。</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-number">5</div><div class="metric-label">核心模块</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-number">2+</div><div class="metric-label">支持类别</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-number">KNN</div><div class="metric-label">最近邻分类</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-number">SGD</div><div class="metric-label">优化可视化</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 项目完成内容
        - Least Squares Linear Regression 示例与矩阵计算过程
        - KNN 图像分类，支持不同 K 值对比
        - 用户上传图片预测
        - 线性分类器训练与模板图像可视化
        - SGD / Momentum 梯度下降路径对比
        - MSE / SVM / Softmax Loss 计算过程演示
        """)
    with col2:
        st.markdown("""
        ### 推荐数据集格式
        ```text
        class4/
        ├── Knife/
        │   ├── 001.jpg
        │   └── ...
        └── Pistol/
            ├── 001.jpg
            └── ...
        ```
        GitHub 部署时可以放在 `data/class4/`，也可以在网页中上传 zip。
        """)

    st.info("左侧选择模块开始实验。没有上传真实数据时，系统会使用自动生成的示例数据保证功能可运行。")


def show_deploy():
    st.markdown('<div class="main-title">GitHub + Streamlit 部署说明</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 1. 项目文件结构
    ```text
    your_project/
    ├── app.py
    ├── requirements.txt
    └── data/
        └── class4/
            ├── Knife/
            └── Pistol/
    ```

    ### 2. requirements.txt
    ```text
    streamlit
    numpy
    pandas
    matplotlib
    Pillow
    ```

    ### 3. 本地运行
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ### 4. Streamlit Cloud 发布
    - 把项目上传到 GitHub
    - 登录 Streamlit Community Cloud
    - New app
    - 选择你的 GitHub 仓库
    - Main file path 填写：`app.py`
    - Deploy

    ### 5. 数据集建议
    如果图片数量不大，可以直接放进 GitHub 的 `data/class4/`。
    如果图片较多，建议网页运行后使用 zip 上传，避免仓库过大。
    """)


# ============================================================
# 主程序入口
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 Vibe Coding")
    st.caption("图像分类与机器学习可视化平台")
    module = st.radio(
        "选择功能模块",
        [
            "首页 Dashboard",
            "1. Least Squares Linear Regression",
            "2. KNN 图像分类",
            "3. 线性分类器与模板可视化",
            "4. SGD / Momentum 可视化",
            "5. Loss 计算演示",
            "部署说明",
        ],
    )
    st.markdown("---")
    st.caption("数据支持：示例数据 / 本地 data/class4 / 上传 zip / 上传单张图片")

if module == "首页 Dashboard":
    show_home()
elif module == "1. Least Squares Linear Regression":
    show_regression()
elif module == "2. KNN 图像分类":
    show_knn()
elif module == "3. 线性分类器与模板可视化":
    show_linear_classifier()
elif module == "4. SGD / Momentum 可视化":
    show_gradient_descent()
elif module == "5. Loss 计算演示":
    show_loss_demo()
else:
    show_deploy()
