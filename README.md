# LSTM Trading Bot for MetaTrader 5

This trading bot leverages a Long Short-Term Memory (LSTM) neural network to predict High, Low, and Close (HLC) values for financial instruments and executes trades based on these predictions.

## Features

- Connects to MetaTrader 5 to fetch historical data
- Trains an LSTM model to predict HLC values
- Automatically executes trades based on predictions
- Sends Telegram notifications for trade monitoring
- Periodically retrains the model for optimal performance
- Generates performance metrics and visualizations
- Manages orders with automatic Stop Loss and Take Profit levels
- Handles MT5 data structures intelligently, including tick volume mapping

## Requirements

To run this bot, you'll need:

1. Python 3.7 or higher
2. MetaTrader 5 installed
3. A MetaTrader 5 account (demo or live)
4. A Telegram bot (for notifications)

## Installation

1. Clone or download this repository
2. Run the bot with automatic dependency installation:

```bash
python run-bot.py
```

This will automatically install all required dependencies from the requirements.txt file.

## Configuration

The `config.json` file contains all parameters needed to configure the bot:

### MT5 Credentials
```json
"mt5_credentials": {
    "login": 11111111111,
    "password": "your_password",
    "server": "YourBroker-Demo",
    "path": "C:\\Program Files\\Your MT5 Terminal Path\\terminal64.exe"
}
```

### Trading Parameters
```json
"symbol": "EURUSD",
"timeframe": "H1",
"look_back": 120,
"retraining_hours": 24,
```

### Order Parameters
```json
"lot_size": 0.01,
"tp_multiplier": 2.5,
"sl_multiplier": 1.0,
"trailing_start_pct": 0.4,
"trailing_step_pct": 0.1,
"risk_per_trade_pct": 1.0,
```

### Telegram Configuration
```json
"telegram_bot_token": "your_telegram_bot_token",
"telegram_chat_id": "your_chat_id",
```

### Model Parameters
```json
"confidence_threshold": 0.75,
"price_change_threshold": 0.4,
"max_data_points": 50000
```

## Directory Independence

This bot has been designed to run correctly regardless of which directory you execute it from. It automatically:

- Finds its own location
- Locates the config.json file
- Saves all output files to the script directory

## How It Works

1. The bot connects to MetaTrader 5 and downloads historical data
2. It processes the data, automatically mapping MT5's 'tick_volume' to 'volume' if needed
3. The LSTM model is trained to predict future HLC values
4. For each candle, the model makes a prediction
5. If the predicted price change exceeds the configured threshold, it executes a trade
6. Stop Loss and Take Profit levels are calculated based on ATR (Average True Range)
7. Notifications about predictions and trades are sent via Telegram
8. The model is periodically retrained according to the configuration

## Metrics and Evaluation

The bot generates and saves:
- Loss curves during training
- Visualizations of predictions vs actual values
- Evolution of metrics over time (MSE, MAE, R²)

These visualizations are saved in the script directory and also sent to Telegram for easy monitoring.

## Error Handling

The bot includes robust error handling that:
- Adapts to different MetaTrader 5 data structures
- Provides clear error messages when issues occur
- Continues operation after recoverable errors
- Automatically attempts to reconnect after lost connections

## Precautions

- This bot is for educational and research purposes
- Use a demo account first to test performance
- Algorithmic trading involves risks; trade with caution
- Profits are not guaranteed; conduct your own risk assessment

## Customization

You can modify:
- The LSTM model architecture in the `build_model()` function
- The trading strategy in the `place_order()` function
- Metrics and visualizations according to your needs
- Price change thresholds to suit your trading style

## Potential Improvements

- Implement more technical indicators as features
- Add market sentiment analysis
- Automatically optimize hyperparameters
- Implement backtesting to validate strategies
- Add more risk management strategies

## Troubleshooting

If you encounter issues:
1. Check the log file (trading_bot.log) for detailed error information
2. Verify your MetaTrader 5 installation and connection details
3. Ensure your Telegram bot token and chat ID are correct
4. Make sure you have sufficient permissions to write to the script directory