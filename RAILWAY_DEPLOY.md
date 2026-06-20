# Railway Deployment Guide

هذا المشروع فيه **service تانية** لازم تتعمل deploy منفصلة على Railway.

---

## الخطوات

### 1️⃣ ابدأ بـ nlp-service (لأن api-gateway بيحتاجها)

1. افتح [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. اختار الـ repo → غيّر **Root Directory** إلى: `nlp-service`
3. Railway هيلاقي الـ `Dockerfile` تلقائياً
4. في **Variables** أضف:
   ```
   SUMMARIZATION_PROVIDER=huggingface
   HUGGINGFACE_MODEL=Falconsai/text_summarization
   LOG_LEVEL=info
   MAX_INPUT_LENGTH=50000
   ```
5. انتظر الـ deploy (ممكن ياخد 5-10 دقايق أول مرة بسبب تحميل PyTorch)
6. انسخ الـ URL اللي هيظهر، مثلاً: `https://nlp-service-xxxx.railway.app`

---

### 2️⃣ بعدين api-gateway

1. في نفس الـ project → **New Service** → **GitHub Repo** → غيّر **Root Directory** إلى: `api-gateway`
2. في **Variables** أضف:
   ```
   NLP_SERVICE_URL=https://nlp-service-xxxx.railway.app   ← الـ URL من فوق
   NODE_ENV=production
   MAX_FILE_SIZE_MB=10
   REQUEST_TIMEOUT_MS=120000
   ALLOWED_ORIGINS=*
   RATE_LIMIT_WINDOW_MS=60000
   RATE_LIMIT_MAX=30
   API_KEY=
   ```
3. بعد الـ deploy، الـ api-gateway URL هو اللي هتستخدمه

---

## ⚠️ ملاحظات مهمة

- **لا ترفع الاتنين في نفس الوقت** — الـ api-gateway هيفشل لو الـ nlp-service لسه مش شغال
- **Railway PORT**: كلا الـ Dockerfiles بيستخدم `${PORT:-default}` عشان Railway بيحدد الـ PORT تلقائياً
- **HuggingFace model**: أول deploy بيحمّل الـ model (~500MB) — ده طبيعي
- **Health check**: Railway بيتحقق من `/health` — الـ endpoint موجود في كلا الـ services

---

## روابط مفيدة

- nlp-service docs: `https://your-nlp.railway.app/docs`
- api-gateway docs: `https://your-gateway.railway.app/api/v1/docs`
