# Bot Monitoring Setup Guide

## Overview
The bot now includes a **monitoring/logging feature** that sends activity logs to a secondary Telegram bot. This allows you to track all user interactions and scheduled tasks on a private monitoring channel.

## Setup Instructions

### Step 1: Create Secondary Bot (if you haven't already)
1. Talk to [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot and get its **API token**
3. Start the bot so it's active
4. Create a new Telegram group or use a private channel for monitoring logs
5. Get your **Chat ID**:
   - Send a message to the bot in your monitoring chat
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Look for the `"chat": {"id": <YOUR_CHAT_ID>}` in the JSON response
   - Save this Chat ID

### Step 2: Configure Environment Variables
Add the following to your `.env` file:

```env
# Secondary Bot for Monitoring/Logging
SECONDARY_BOT_TOKEN=8743935123:AAEei2m8MeOBxT5Js7wojtxYfcbtxmSHpPE
MONITOR_CHAT_ID=<YOUR_CHAT_ID>
```

**Replace:**
- `8743935123:AAEei2m8MeOBxT5Js7wojtxYfcbtxmSHpPE` with your secondary bot's actual token
- `<YOUR_CHAT_ID>` with your actual monitoring chat ID (e.g., `-123456789`)

### Step 3: Restart Your Bot
Once you've updated the `.env` file, restart your main crypto bot:

```bash
python bot.py
```

## What Gets Logged

### User Activity Logs
✅ **Analysis Requests** - Every `/analyze <coin>` command
- User ID and username
- Cryptocurrency symbol analyzed
- Timestamp

✅ **Follow-up Questions** - Every message after an analysis
- User asking the question
- Symbol being discussed
- First 100 characters of the query

✅ **Data Downloads** - Every `/data` command
- User requesting export
- Cryptocurrency symbol

✅ **Commands Used**
- `/start` - User starts bot
- `/help` - User requests help
- `/about` - User views about info
- `/renew` - User resets chat
- `/train` - User manually triggers DRL training

### Scheduled Task Logs
🤖 **DRL Model Training** (runs daily at 00:30 UTC / 07:30 Vietnam time)
- **Training Started** - Logs when scheduled task begins
- **Training Completed** - Logs successful completion with model path
- **Training Failed** - Logs any errors

All logs include:
- ⏰ **Timestamp** in Vietnam timezone (Asia/Ho_Chi_Minh)
- 👤 **User information** (ID, username)
- 💱 **Crypto symbol** (when applicable)
- 📊 **Activity details**

## Message Format Example

```
⏰ 14:32:15 01/03/2026
📊 Analysis Request:
👤 User: @john_doe (ID: 123456789)
💱 Symbol: BTC
```

## Troubleshooting

### Logs not appearing?
1. **Check environment variables** - Make sure `SECONDARY_BOT_TOKEN` and `MONITOR_CHAT_ID` are set correctly
   ```bash
   echo $SECONDARY_BOT_TOKEN
   echo $MONITOR_CHAT_ID
   ```

2. **Verify bot token** - Test the token works:
   ```bash
   curl https://api.telegram.org/bot<TOKEN>/getMe
   ```

3. **Check chat ID** - Make sure the bot has permission to send messages in that chat
   - The monitoring chat should be private or have the bot as a member
   - Try manually sending a test message with the bot

4. **Check logs** - Look at bot console output for any errors:
   ```
   Failed to send monitor log: ...
   ```

5. **Test send** - You can test by manually triggering an analysis command:
   ```bash
   # In Telegram, send: /analyze BTC
   ```

### Bot won't send logs but says it's configured?
- Make sure the secondary bot is actually running/active
- Verify the Chat ID is correct (format: negative number for groups, e.g., `-123456789`)
- Check that there are no API rate limits

## Disabling Monitoring
If you want to disable monitoring logs temporarily:
1. Remove or comment out `SECONDARY_BOT_TOKEN` and `MONITOR_CHAT_ID` from `.env`
2. Restart the bot
3. Logs will be skipped silently (check logs for: "Monitoring bot not configured")

## Security Notes
- Keep your secondary bot token **private** - don't share it in code or version control
- Use a private monitoring channel - only you should have access
- Each log message contains user IDs and usernames - ensure your monitoring chat is secure
- Logs are sent only within your private bot infrastructure

## Integration Example
You can use monitoring logs for:
- **User analytics** - Track which coins are most analyzed
- **System monitoring** - Track DRL training completion times
- **Alerts** - Set up notifications for failed training jobs
- **Auditing** - Keep records of all bot usage

---

**Created:** March 1, 2026
**Last Updated:** March 1, 2026
