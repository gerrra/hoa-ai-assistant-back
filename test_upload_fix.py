#!/usr/bin/env python3
import requests

# Тест загрузки документа после исправления таблицы topics
print("1. Аутентификация...")
session = requests.Session()
login_response = session.post("http://localhost:8000/admin/api/login", 
                             json={"password": "AaBb@@1122!!"})

if login_response.status_code != 200:
    print("❌ Ошибка аутентификации")
    print("Response:", login_response.text)
    exit(1)

print("✅ Аутентификация успешна")

print("2. Загружаем документ...")
test_file = ("test.txt", "Тест после исправления таблицы topics", "text/plain")
upload_response = session.post("http://localhost:8000/admin/api/upload",
                             files={"file": test_file},
                             data={
                                 "community_id": 1,
                                 "title": "Тест после исправления таблицы topics",
                                 "doc_type": "regulation",
                                 "visibility": "resident",
                                 "use_topic_analysis": "true"
                             })

print(f"   Status: {upload_response.status_code}")
print(f"   Response: {upload_response.text}")

if upload_response.status_code == 200:
    print("✅ Документ загружен успешно!")
    print("🎉 Проблема с таблицей topics решена!")
else:
    print("❌ Ошибка загрузки")
    print("Проверяем логи...")
    import subprocess
    result = subprocess.run([
        "docker", "logs", "hoa-db", "--tail", "5"
    ], capture_output=True, text=True)
    print("Логи БД:")
    print(result.stdout)