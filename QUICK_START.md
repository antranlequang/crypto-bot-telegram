# Quick Start: Bot Monitoring Feature

## What Was Added
Your crypto bot now has **real-time activity monitoring** that sends logs to your secondary Telegram bot.

## Setup (2 Simple Steps)

### 1️⃣ Update `.env` File
Add these lines to your `.env`:
```
SECONDARY_BOT_TOKEN=8743935123:AAEei2m8MeOBxT5Js7wojtxYfcbtxmSHpPE
MONITOR_CHAT_ID=<your_chat_id_here>
```

**How to get `MONITOR_CHAT_ID`:**
1. Send a message to your secondary bot in your monitoring chat
2. Open: `https://api.telegram.org/bot8743935123:AAEei2m8MeOBxT5Js7wojtxYfcbtxmSHpPE/getUpdates`
3. Find `"chat": {"id": <number>}` → that's your Chat ID
4. Replace `<your_chat_id_here>` with it (example: `-123456789`)

### 2️⃣ Restart Bot
```bash
python bot.py
```

## What Gets Monitored

| Activity | Log Message | When |
|----------|-------------|------|
| 🚀 Bot Start | User joins | `/start` command |
| 📊 Analysis | Coin analyzed | `/analyze BTC` command |
| 💬 Questions | Follow-up query | User asks about analysis |
| 📰 Help | Help requested | `/help` command |
| 💾 Data Export | Data downloaded | `/data` command |
| 🤖 DRL Training | Model retraining | Daily at 00:30 UTC |
| ✅ Training Done | Model complete | After training succeeds |
| ❌ Training Failed | Error occurred | Training fails |

## Example Logs You'll See

```
⏰ 14:32:15 01/03/2026
📊 Analysis Request:
👤 User: @john_doe (ID: 123456789)
💱 Symbol: BTC
```

```
⏰ 00:30:01 02/03/2026
✅ Scheduled DRL Training Completed:
📊 BTC data updated
🤖 Model retrained and saved to models/drl_ppo_btc_2026_03_02.zip
```

## Code Changes Summary

### Files Modified:
1. **config.py** - Added secondary bot token variables
2. **bot.py** - Added monitoring function and logging calls

### New Function:
- `send_to_monitor_bot(message)` - Sends messages to secondary bot with Vietnam timezone

### Logged Handlers:
- `start()` - User starts bot
- `help_command()` - Help requested
- `about_command()` - About info viewed
- `renew_command()` - Chat renewed
- `train_command()` - Manual DRL training
- `analyze_command()` - Analysis requested
- `analyze_text()` - Follow-up questions
- `data_command()` - Data export
- `daily_btc_update()` - Scheduled DRL training

## Troubleshooting

**Not receiving logs?**
1. Check `.env` file has correct token and Chat ID
2. Verify secondary bot is active and member of monitoring chat
3. Check main bot console for errors: `Failed to send monitor log`
4. Test token: `curl https://api.telegram.org/bot<token>/getMe`

**Want to disable?**
- Remove `SECONDARY_BOT_TOKEN` and `MONITOR_CHAT_ID` from `.env`
- Logs will be skipped silently

## Additional Files
- **MONITORING_SETUP.md** - Detailed setup guide with troubleshooting
- **This file** - Quick reference

---

**Ready to monitor!** 🚀 Your logs will appear in the secondary bot as users interact with the main bot.
