const HELP_TEXT = [
  "BIST bot komutlari:",
  "",
  "/run - son 7 gunu cekip bugunku raporu uretir",
  "/run 2026-06-07 - secilen gun icin rapor uretir",
  "/help - komutlari gosterir",
].join("\n");

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {"content-type": "application/json; charset=utf-8"},
  });
}

function parseCommand(text) {
  const parts = text.trim().split(/\s+/);
  const command = (parts[0] || "").split("@")[0].toLowerCase();
  const date = parts[1] || "";
  return {command, date};
}

function validateDate(date) {
  return /^\d{4}-\d{2}-\d{2}$/.test(date);
}

function allowedChatIds(env) {
  return (env.TELEGRAM_ALLOWED_CHAT_ID || "")
    .split(",")
    .map((chatId) => chatId.trim())
    .filter(Boolean);
}

async function sendTelegram(env, chatId, text) {
  const body = new URLSearchParams({
    chat_id: chatId,
    text,
  });
  await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: {"content-type": "application/x-www-form-urlencoded"},
    body,
  });
}

function requireEnv(env, names) {
  const missing = names.filter((name) => !env[name]);
  if (missing.length) {
    throw new Error(`Eksik Cloudflare secret/var: ${missing.join(", ")}`);
  }
}

async function dispatchWorkflow(env, reportDate, telegramChatId) {
  requireEnv(env, [
    "GITHUB_TOKEN",
    "GITHUB_OWNER",
    "GITHUB_REPO",
    "TELEGRAM_BOT_TOKEN",
  ]);

  const workflowFile = env.GITHUB_WORKFLOW_FILE || "daily-pipeline.yml";
  const ref = env.GITHUB_REF || "main";
  const inputs = {
    kap_days: reportDate ? "1" : env.DEFAULT_KAP_DAYS || "7",
    baseline_lookback_days: env.DEFAULT_BASELINE_LOOKBACK_DAYS || "365",
    baseline_min_history: env.DEFAULT_BASELINE_MIN_HISTORY || "5",
    report_date: reportDate || "",
    telegram_chat_id: telegramChatId || "",
  };
  const response = await fetch(
    `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/${workflowFile}/dispatches`,
    {
      method: "POST",
      headers: {
        accept: "application/vnd.github+json",
        authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "content-type": "application/json",
        "user-agent": "bist-telegram-workflow-dispatcher",
        "x-github-api-version": "2022-11-28",
      },
      body: JSON.stringify({ref, inputs}),
    },
  );
  if (response.status !== 204) {
    const body = await response.text();
    throw new Error(`GitHub workflow tetiklenemedi: HTTP ${response.status} ${body}`);
  }
  return {ref, inputs};
}

export default {
  async fetch(request, env) {
    if (request.method === "GET") {
      return new Response(HELP_TEXT, {headers: {"content-type": "text/plain; charset=utf-8"}});
    }
    if (request.method !== "POST") {
      return json({ok: false, error: "method_not_allowed"}, 405);
    }

    const secret = request.headers.get("x-telegram-bot-api-secret-token") || "";
    if (env.TELEGRAM_WEBHOOK_SECRET && secret !== env.TELEGRAM_WEBHOOK_SECRET) {
      return json({ok: false, error: "unauthorized"}, 401);
    }

    const update = await request.json();
    const message = update.message || update.edited_message;
    const chatId = message?.chat?.id ? String(message.chat.id) : "";
    const text = message?.text || "";
    if (!chatId || !text) {
      return json({ok: true, ignored: true});
    }

    const allowedIds = allowedChatIds(env);
    if (allowedIds.length && !allowedIds.includes(chatId)) {
      await sendTelegram(env, chatId, `Bu bot icin yetkin yok.\nBu chat ID'yi izin listesine ekle: ${chatId}`);
      return json({ok: true, unauthorized_chat: true});
    }

    const {command, date} = parseCommand(text);
    if (command === "/start" || command === "/help") {
      await sendTelegram(env, chatId, HELP_TEXT);
      return json({ok: true});
    }
    if (command !== "/run") {
      await sendTelegram(env, chatId, "Bilinmeyen komut.\n\n" + HELP_TEXT);
      return json({ok: true, unknown_command: command});
    }
    if (date && !validateDate(date)) {
      await sendTelegram(env, chatId, "Tarih formati gecersiz. Ornek: /run 2026-06-07");
      return json({ok: true, invalid_date: date});
    }

    try {
      const result = await dispatchWorkflow(env, date, chatId);
      const suffix = date ? `Rapor tarihi: ${date}` : "Rapor tarihi: son veri gunu";
      await sendTelegram(env, chatId, `Workflow tetiklendi.\n${suffix}\nRef: ${result.ref}`);
      return json({ok: true, dispatched: result});
    } catch (error) {
      await sendTelegram(env, chatId, `Workflow tetiklenemedi: ${error.message}`);
      return json({ok: false, error: error.message}, 500);
    }
  },
};
