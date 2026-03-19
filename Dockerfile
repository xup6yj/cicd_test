# 使用官方 Python image
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製檔案
COPY . .

# 執行程式
CMD ["python", "app.py"]