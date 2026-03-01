#!/usr/bin/env python3
"""
DRL Bitcoin Trading Model - Standalone Python Script
Trains a Deep Reinforcement Learning (PPO) agent to trade Bitcoin

Usage:
    python drl_model.py                 # Run with all defaults
    python drl_model.py --help          # Show help
    python drl_model.py --seed 42       # Use custom seed
    python drl_model.py --timesteps 500000  # Train longer
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import io
import random
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import bot config, fallback to defaults if not available
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PPO_BTC_MODEL_PATH
    MODEL_PATH = PPO_BTC_MODEL_PATH
except (ImportError, AttributeError):
    MODEL_PATH = "models/drl_ppo_btc.zip"
    logger.warning(f"Bot config not found. Using default model path: {MODEL_PATH}")


class BitcoinTradingEnv(gym.Env):
    """Custom trading environment for Bitcoin using PPO agent"""
    
    def __init__(self, df, initial_balance=10000, window_size=20):
        """
        Initialize the trading environment
        
        Args:
            df: DataFrame with OHLCV and indicator data
            initial_balance: Starting capital in USD
            window_size: Number of past days to observe
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_features = df.shape[1]

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized price history
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * self.n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.btc_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.trades = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """Get normalized observation from price window"""
        window = self.df.iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        # Normalize
        obs = (window - mean) / std
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs.flatten().astype(np.float32)

    def step(self, action):
        """Execute one trading action"""
        price = float(self.df.iloc[self.current_step]['close'])

        # Guard against invalid prices
        if price <= 0 or np.isnan(price):
            price = 1e-8

        prev_worth = self.net_worth

        # Execute action
        if action == 1 and self.balance >= price:  # Buy
            btc_bought = (self.balance * 0.95) / price
            self.btc_held += btc_bought
            self.balance -= btc_bought * price
            self.trades += 1
        elif action == 2 and self.btc_held > 0:  # Sell
            self.balance += self.btc_held * price
            self.btc_held = 0.0
            self.trades += 1

        # Update portfolio value
        self.net_worth = self.balance + self.btc_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate reward
        profit = (self.net_worth - prev_worth) / (abs(prev_worth) + 1e-8)
        drawdown = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-8)
        reward = float(np.clip(profit * 100 - drawdown * 10, -10, 10))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_obs(), reward, done, False, {
            'net_worth': self.net_worth,
            'balance': self.balance
        }


def load_data(data_source='local', google_drive_id=None):
    """
    Load training data from local file or Google Drive
    
    Args:
        data_source: 'local' or 'gdrive'
        google_drive_id: Google Drive file ID if using gdrive source
        
    Returns:
        DataFrame with processed data
    """
    logger.info("Loading data...")
    
    if data_source == 'local':
        local_csv = "btc_features.csv"
        if os.path.exists(local_csv):
            logger.info(f"Loading local data from {local_csv}")
            data_raw = pd.read_csv(local_csv)
        else:
            logger.error(f"Local file {local_csv} not found")
            logger.info("Trying Google Drive...")
            data_source = 'gdrive'
    
    if data_source == 'gdrive':
        google_drive_id = google_drive_id or '10XOY2JXYuRpoVTCZhO9nVouaueK-CjoB'
        logger.info(f"Downloading from Google Drive (ID: {google_drive_id})")
        
        download_url = f'https://drive.google.com/uc?export=download&id={google_drive_id}'
        try:
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            data_raw = pd.read_csv(io.StringIO(response.text))
            logger.info("✅ Data downloaded from Google Drive")
        except Exception as e:
            logger.error(f"Failed to download from Google Drive: {e}")
            sys.exit(1)
    
    # Rename columns if in Vietnamese
    column_mapping = {
        'Ngày': 'Date',
        'Lần cuối': 'Close',
        'Mở': 'Open',
        'Cao': 'High',
        'Thấp': 'Low',
        'KL': 'Volume',
        '% Thay đổi': 'Change_Percent',
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
    }
    
    data_raw = data_raw.rename(columns=column_mapping)
    
    logger.info(f"Data shape: {data_raw.shape}")
    logger.info(f"Columns: {list(data_raw.columns)}")
    
    return data_raw


def prepare_data(data_raw):
    """
    Select required columns and validate data
    
    Args:
        data_raw: Raw DataFrame
        
    Returns:
        Processed DataFrame, scaled DataFrames for train/test
    """
    logger.info("Preparing data...")
    
    required_cols = ['close', 'open', 'high', 'low', 'vol', 'rsi_14', 
                     'macd_line_6_20', 'macd_signal_6_20', 'roc_12', 
                     'atr_14', 'std_dev_20', 'obv']
    
    # Normalize column names to lowercase
    data_raw.columns = [col.lower() for col in data_raw.columns]
    
    # Check which columns are available
    available_cols = [col for col in required_cols if col in data_raw.columns]
    missing_cols = [col for col in required_cols if col not in data_raw.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.warning(f"Using available columns: {available_cols}")
        data_final = data_raw[available_cols]
    else:
        data_final = data_raw[required_cols]
    
    logger.info(f"Selected {len(data_final.columns)} columns for training")
    
    # Scale data using RobustScaler (better for finance)
    logger.info("Scaling data with RobustScaler...")
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(data_final),
        columns=data_final.columns,
        index=data_final.index
    )
    
    # Handle NaN/Inf values
    df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Data completeness: {(1 - df_scaled.isna().sum().sum() / (df_scaled.shape[0] * df_scaled.shape[1])) * 100:.1f}%")
    
    # Train-test split (80-20)
    split = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:split].reset_index(drop=True)
    test_df = df_scaled.iloc[split:].reset_index(drop=True)
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return data_final, train_df, test_df


def train_model(train_df, seed=12, timesteps=300000):
    """
    Train PPO agent on training data
    
    Args:
        train_df: Training DataFrame
        seed: Random seed for reproducibility
        timesteps: Number of training timesteps
        
    Returns:
        Trained model
    """
    logger.info(f"Training configuration:")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Timesteps: {timesteps}")
    logger.info(f"  Initial Balance: $10,000")
    logger.info(f"  Window Size: 20 days")
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    
    logger.info("Creating training environment...")
    train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, initial_balance=10000, window_size=20)])
    
    logger.info("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 128])
    )
    
    logger.info(f"Training model for {timesteps:,} timesteps...")
    logger.info("This may take several minutes...")
    
    model.learn(total_timesteps=timesteps)
    
    return model


def save_model(model, model_path=None):
    """
    Save trained model to disk
    
    Args:
        model: Trained PPO model
        model_path: Path to save model (uses config default if not specified)
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    logger.info(f"Saving model to {model_path}...")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Remove .zip extension if present (stable-baselines3 adds it automatically)
    save_path = model_path.replace('.zip', '')
    
    model.save(save_path)
    logger.info(f"✅ Model saved to {model_path}")


def evaluate_model(model, test_df):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PPO model
        test_df: Test DataFrame
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info("Evaluating model on test set...")
    
    env = BitcoinTradingEnv(test_df)
    obs, _ = env.reset()
    net_worths = [env.initial_balance]
    
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        net_worths.append(info['net_worth'])
        steps += 1
    
    net_worths = np.array(net_worths)
    
    # Calculate metrics
    final_worth = net_worths[-1]
    cumulative_return = (final_worth - net_worths[0]) / net_worths[0] * 100
    
    daily_returns = np.diff(net_worths) / (net_worths[:-1] + 1e-8)
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(365)
    
    rolling_max = np.maximum.accumulate(net_worths)
    drawdowns = (rolling_max - net_worths) / (rolling_max + 1e-8)
    max_drawdown = drawdowns.max() * 100
    
    initial_value = net_worths[0]
    num_days = len(net_worths) - 1
    total_years = max(num_days / 365.0, 0.01)
    cagr = ((final_worth / initial_value) ** (1 / total_years)) - 1
    
    calmar_ratio = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0
    
    metrics = {
        'final_net_worth': final_worth,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'total_trades': env.trades,
        'steps': steps
    }
    
    # Log results
    logger.info("\n" + "="*50)
    logger.info("📊 PERFORMANCE METRICS")
    logger.info("="*50)
    logger.info(f"Final Net Worth:      ${final_worth:,.2f}")
    logger.info(f"Cumulative Return:    {cumulative_return:.2f}%")
    logger.info(f"Sharpe Ratio:         {sharpe_ratio:.4f}")
    logger.info(f"Max Drawdown:         {max_drawdown:.2f}%")
    logger.info(f"Calmar Ratio:         {calmar_ratio:.4f}")
    logger.info(f"Total Trades:         {env.trades}")
    logger.info("="*50 + "\n")
    
    return metrics, net_worths


def plot_results(test_df, net_worths):
    """
    Plot training results
    
    Args:
        test_df: Test DataFrame with close prices
        net_worths: Array of portfolio values over time
    """
    logger.info("Plotting results...")
    
    # Buy and hold baseline
    buy_hold = test_df['close'].values / test_df['close'].values[0] * 10000
    
    # Plot 1: Agent vs Buy & Hold
    plt.figure(figsize=(14, 5))
    plt.plot(net_worths, label='PPO Agent', color='blue', linewidth=2)
    plt.plot(buy_hold[:len(net_worths)], label='Buy & Hold', color='orange', linestyle='--', linewidth=2)
    plt.axhline(10000, color='gray', linestyle=':', label='Initial Balance', linewidth=1)
    plt.title('PPO Bitcoin Trading Agent vs Buy & Hold', fontsize=14, fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('drl_performance.png', dpi=150)
    logger.info("Saved plot: drl_performance.png")
    
    # Plot 2: Bitcoin close price
    plt.figure(figsize=(14, 5))
    plt.plot(test_df['close'].values, label='Close Price', color='green', linewidth=1.5)
    plt.title('Bitcoin Close Price (Test Period)', fontsize=14, fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Normalized Price')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('btc_price_test.png', dpi=150)
    logger.info("Saved plot: btc_price_test.png")
    
    plt.show()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='DRL Bitcoin Trading Model Training')
    parser.add_argument('--seed', type=int, default=12, help='Random seed (default: 12)')
    parser.add_argument('--timesteps', type=int, default=200000, help='Training timesteps (default: 200000)')
    parser.add_argument('--source', choices=['local', 'gdrive'], default='local', help='Data source')
    parser.add_argument('--gdrive-id', type=str, help='Google Drive file ID')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--model-path', type=str, help='Custom model save path')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🤖 DRL BITCOIN TRADING MODEL - TRAINING SCRIPT")
    logger.info("="*60)
    
    try:
        # Load data
        data_raw = load_data(args.source, args.gdrive_id)
        
        # Prepare data
        data_final, train_df, test_df = prepare_data(data_raw)
        
        # Train model
        model = train_model(train_df, seed=args.seed, timesteps=args.timesteps)
        
        # Save model
        save_model(model, args.model_path or MODEL_PATH)
        
        # Evaluate model
        metrics, net_worths = evaluate_model(model, test_df)
        
        # Plot results
        if not args.no_plot:
            plot_results(test_df, net_worths)
        
        logger.info("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
