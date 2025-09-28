import requests
import json

# ========== 設定 ==========
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "menu-bot"
ORDERS_JSONL = "orders.jsonl"
# ==========================

# 載入測試資料


def load_test_orders():
    orders = []
    with open(ORDERS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                order_data = json.loads(line.strip())
                orders.append(order_data)
            except json.JSONDecodeError:
                continue
    return orders[:100]


def test_order(prompt, expected):
    # 呼叫 Ollama API
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4]
                            },
                            "name": {
                                "type": "string",
                                "enum": ["紅茶", "綠茶", "烏龍茶", "四季春"]
                            },
                            "count": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "size": {
                                "type": "string",
                                "enum": ["中", "大"]
                            },
                            "sweetness": {
                                "type": "string",
                                "enum": ["無糖", "9分甜", "8分甜", "少糖", "6分甜", "半糖", "4分甜", "微糖", "2分甜", "1分甜", "正常甜"]
                            },
                            "temperature": {
                                "type": "string",
                                "enum": ["正常冰", "少冰", "微冰", "去冰", "常溫", "溫", "熱"]
                            },

                            "addons": {
                                "type": "array",
                                "items": {
                                        "type": "string",
                                        "enum": ["珍珠", "波霸", "椰果", "混珠", "珍波椰", "珍椰", "波椰"]
                                }
                            }

                        },
                        "required": [
                            "id",
                            "name",
                            "count",
                            "size",
                            "sweetness",
                            "temperature",
                            "addons"
                        ]
                    }
                }
            },
            "required": [
                "items"
            ]
        }
    }

    try:
        resp = requests.post(OLLAMA_API, json=payload)
        resp.raise_for_status()
        data = resp.json()

        print(f"\n測試訂單: {prompt}")
        print("模型回應:", data["response"])

        # 嘗試解析模型回應
        try:
            order = json.loads(data["response"])
            print("✅ 模型輸出成功解析成 JSON")
        except json.JSONDecodeError as e:
            print("❌ 模型輸出不是合法 JSON:", e)
            return False

        # 比對結構和內容
        if order == expected:
            print("✅ 完全匹配")
            return True
        else:
            print("❌ 內容不匹配")
            print("預期:", json.dumps(expected, ensure_ascii=False))
            print("實際:", json.dumps(order, ensure_ascii=False))
            return False

    except Exception as e:
        print(f"❌ API 呼叫失敗: {e}")
        return False


def main():
    test_orders = load_test_orders()
    print(f"載入了 {len(test_orders)} 筆測試訂單")

    success = 0
    for test_case in test_orders:
        prompt = test_case["prompt"]
        expected = test_case["completion"]
        if test_order(prompt, json.loads(expected)):
            success += 1

    print(f"\n測試完成: {success}/{len(test_orders)} 筆成功")


if __name__ == "__main__":
    main()
