import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Slurry Optimization", layout="wide")
st.title("🔬 Anode Slurry 조성 최적화 (GP vs RF 모델 비교)")

# 데이터 불러오기
df = pd.read_csv("slurry_data.csv")
x_cols = ["carbon_black_g", "graphite_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# 텐서 변환
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

if st.button("🔍 조성 추천 및 모델 예측 비교 보기"):

    # GP 모델 학습
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # RF 모델 학습
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, Y_raw.ravel())

    # 최적 조성 탐색
    best_y = train_y.max()
    bounds = torch.stack([
        torch.zeros(train_x.shape[1], dtype=torch.double),
        torch.ones(train_x.shape[1], dtype=torch.double)
    ])
    acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)

    candidate_scaled, _ = optimize_acqf(
        acq_function=acq_fn,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20
    )
    candidate_np = candidate_scaled.detach().numpy()
    candidate_original = x_scaler.inverse_transform(candidate_np)

    st.subheader("추천된 조성 (단위: g)")
    for i, col in enumerate(x_cols):
        st.write(f"{col}: **{candidate_original[0][i]:.3f} g**")

    # 예측 및 비교
    carbon_black_index = x_cols.index("carbon_black_g")
    x_vals = np.linspace(0, 1, 100)
    x_test_full = np.tile(np.mean(X_scaled, axis=0), (100, 1))
    x_test_full[:, carbon_black_index] = x_vals

    X_test = torch.tensor(x_test_full, dtype=torch.double)
    model.eval()
    with torch.no_grad():
        pred = model.posterior(X_test)
        gp_mean = pred.mean.numpy().flatten()
        gp_std = pred.variance.sqrt().numpy().flatten()

    rf_mean = rf_model.predict(x_test_full)

    x_vals_g = x_scaler.inverse_transform(x_test_full)[:, carbon_black_index]
    train_cb = X_raw[:, carbon_black_index]
    train_y_np = Y_raw.flatten()

    # RMSE 계산
    with torch.no_grad():
        train_pred_gp = model.posterior(train_x).mean.numpy().flatten()
    train_pred_rf = rf_model.predict(X_scaled)
    rmse_gp = np.sqrt(mean_squared_error(train_y_np, train_pred_gp))
    rmse_rf = np.sqrt(mean_squared_error(train_y_np, train_pred_rf))

    # Feature Importance
    importances = rf_model.feature_importances_
    st.subheader("RF Feature Importance")
    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.bar(x_cols, importances,width=0.4)
    ax_imp.set_ylabel("Importance")
    ax_imp.set_title("Random Forest Feature Importance")
    ax_imp.set_xticklabels(x_cols, rotation=45, ha="right")
    st.pyplot(fig_imp)

    # 예측 그래프 출력
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vals_g, gp_mean, color="blue", label="GP Predicted Mean")
    ax.fill_between(x_vals_g, gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std,
                    color="blue", alpha=0.2, label="GP 95% Confidence Interval")
    ax.plot(x_vals_g, rf_mean, color="green", linestyle="--", label="RF Predicted Mean")
    ax.scatter(train_cb, train_y_np, color="red", label="Observed Data", zorder=10)
    ax.set_title("GP vs RF Prediction")
    ax.set_xlabel("Carbon Black [g]")
    ax.set_ylabel("Yield Stress [Pa]")
    ax.legend()
    ax.grid(True)

    # RMSE 텍스트 박스 추가
    textstr = f"GP RMSE: {rmse_gp:.2f} Pa\nRF RMSE: {rmse_rf:.2f} Pa"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    st.pyplot(fig)
