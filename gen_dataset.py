import json
import random

# 載入菜單
with open("menu.json", "r", encoding="utf-8") as f:
    menu = json.load(f)["menu"]

# 一些常見的點餐語氣模板
TEMPLATES = [
    "我要{count}杯{size}{temperature}{sweetness}{name}{addon}",
    "幫我來{count}杯{size}{sweetness}{temperature}{name}{addon}",
    "點{count}杯{temperature}{sweetness}{size}{name}{addon}謝謝",
    "{count}杯{size}{name}，{sweetness}{temperature}{addon}"
]

def generate_prompt(item, size=None, sweetness=None, temperature=None, addon=None, count=1):
    """組合自然語言點餐"""
    addon_text = f"加{addon}" if addon else ""
    size_text = size if size else ""
    sweet_text = sweetness if sweetness else ""
    temp_text = temperature if temperature else ""
    tpl = random.choice(TEMPLATES)
    return tpl.format(
        count=count,
        size=size_text,
        sweetness=sweet_text,
        temperature=temp_text,
        name=item["name"],
        addon=addon_text
    )

def generate_completion(item, size=None, sweetness=None, temperature=None, addon=None, count=1):
    """組合標準 JSON 訂單"""
    return {
        "items": [
            {
                "id": item["id"],
                "name": item["name"],
                "count": count,
                "size": size,
                "sweetness": sweetness,
                "temperature": temperature,
                "addons": [addon] if addon else []
            }
        ]
    }

# 開始生成資料
dataset = []
for item in menu:
    for size in [s["label"] for s in item.get("size", [])] or [None]:
        for sweet in item.get("sweetness", {}).get("options", [None]):
            for temp in item.get("temperature", {}).get("options", [None]):
                addon_options = [None] + [a["name"] for a in item.get("addons", {}).get("options", [])]
                for addon in addon_options:
                    # 隨機生成 1~2 筆 (避免數據太爆炸)
                    for _ in range(random.randint(1, 2)):
                        count = random.choice([1, 2])
                        prompt = generate_prompt(item, size, sweet, temp, addon, count)
                        completion = generate_completion(item, size, sweet, temp, addon, count)
                        dataset.append({ 
                            "prompt": prompt,
                            "completion": json.dumps(completion, ensure_ascii=False)
                        })

# 存成 jsonl
with open("orders.jsonl", "w", encoding="utf-8") as f:
    for d in dataset:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"✅ 已生成 {len(dataset)} 筆訓練資料到 orders.jsonl")