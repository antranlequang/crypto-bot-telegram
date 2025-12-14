#!/usr/bin/env python3
"""
Utility script to train a lightweight PPO model for BTC regime detection.
The script generates synthetic OHLCV data, computes the same indicators that
the bot uses at inference time, trains a PPO policy on a toy environment, and
exports the weights to models/drl_ppo_btc.zip so the Telegram bot can load it.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

WINDOW_SIZE = 60
DEFAULT_TIMESTEPS = 50_000
MODEL_PATH = os.path.join("models", "drl_ppo_btc.zip")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clone of the indicator pipeline the bot uses. Keeping the same column order
    ensures the PPO model receives identical observations at inference.
    """
    df = df.copy()
    df["Log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["SMA_14"] = df["Close"].rolling(14).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal_Line"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    df.dropna(inplace=True)
    return df


def generate_synthetic_ohlcv(days: int = 1800, start_price: float = 30_000, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic BTC-like OHLCV series using a noisy geometric random walk.
    This keeps the training process reproducible and independent from live APIs.
    """
    rng = np.random.default_rng(seed)
    # Daily log returns with mild drift + volatility
    returns = rng.normal(loc=0.0004, scale=0.02, size=days)
    prices = start_price * np.exp(np.cumsum(returns))
    close_series = pd.Series(prices)
    open_series = close_series.shift(1).fillna(close_series.iloc[0])
    spread = np.abs(rng.normal(0.001, 0.01, size=days))
    open_vals = open_series.to_numpy(dtype=float)
    close_vals = close_series.to_numpy(dtype=float)
    high_vals = np.maximum(open_vals, close_vals) * (1 + spread)
    low_vals = np.minimum(open_vals, close_vals) * np.clip(1 - spread, 0.90, None)
    volume = rng.uniform(1_000, 5_000, size=days)

    df = pd.DataFrame(
        {
            "Open": open_vals,
            "High": high_vals,
            "Low": low_vals,
            "Close": close_vals,
            "Volume": volume,
        },
        index=pd.date_range(end=pd.Timestamp.utcnow(), periods=days, freq="D"),
    )
    return df


@dataclass
class SyntheticBTCDrlEnv(gym.Env):
    """
    Simple long/short environment for PPO that mimics the bot's observation space.
    Actions: 0 = flat, 1 = long, 2 = short.
    Reward = position * return - small trading cost (encourages trend following).
    """

    df_feat: pd.DataFrame
    window_size: int = WINDOW_SIZE

    def __post_init__(self):
        super().__init__()
        self.df_feat = self.df_feat.reset_index(drop=True)
        if len(self.df_feat) <= self.window_size:
            raise ValueError("Not enough rows to build DRL observations.")

        obs_dim = len(self.df_feat.columns) * self.window_size + 2
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._start_idx = self.window_size
        self._idx = self._start_idx
        self.position = 0.0
        self.cash_ratio = 1.0

    def _get_state(self) -> np.ndarray:
        idx = min(self._idx, len(self.df_feat))
        window = self.df_feat.iloc[idx - self.window_size : idx]
        feats = window.values.flatten()
        state = np.concatenate([feats, [self.cash_ratio, self.position]]).astype(np.float32)
        return state

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._idx = self._start_idx
        self.position = 0.0
        self.cash_ratio = 1.0
        return self._get_state(), {}

    def step(self, action: int):
        prev_price = float(self.df_feat["Close"].iloc[self._idx - 1])
        price = float(self.df_feat["Close"].iloc[self._idx])
        ret = (price - prev_price) / max(prev_price, 1e-6)

        prev_pos = self.position
        if action == 1:
            self.position = 1.0
        elif action == 2:
            self.position = -1.0
        else:
            self.position = 0.0

        # Simple position-based reward with transaction cost
        trade_cost = 0.001 * abs(self.position - prev_pos)
        reward = float(self.position * ret - trade_cost)
        self.cash_ratio = max(0.0, 1.0 - 0.5 * abs(self.position))

        self._idx += 1
        terminated = self._idx >= len(self.df_feat)
        obs = self._get_state()
        info = {"position": self.position, "return": ret}
        return obs, reward, terminated, False, info


def train_and_save(total_timesteps: int, seed: int | None = 42):
    print(f"Generating synthetic dataset...")
    df_prices = generate_synthetic_ohlcv()
    df_feat = compute_indicators(df_prices)

    def make_env():
        return SyntheticBTCDrlEnv(df_feat=df_feat, window_size=WINDOW_SIZE)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=None,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved PPO model to {MODEL_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train placeholder DRL PPO model for BTC.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total timesteps for PPO training (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(total_timesteps=args.timesteps, seed=args.seed)
