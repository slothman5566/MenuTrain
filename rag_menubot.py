from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pydantic import BaseModel
from typing import List
import requests
import json
import re
from typing import List, Dict

# ========== 設定 ==========
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "menu-bot"
ORDERS_JSONL = "orders.jsonl"
context_history = []
# ==========================
# ----------------------------
# 定義菜單資料
# ----------------------------
menu_items = [
    "紅茶 大杯L 40 中杯M: 35 溫度必選: 正常冰, 少冰, 微冰, 去冰, 常溫, 溫, 熱 甜度必選:無糖, 9分甜,8分甜, 少糖, 6分甜, 半糖, 4分甜, 微糖,2分甜,1分甜,正常甜 加料可選: 珍珠, 波霸, 椰果, 混珠, 珍波椰, 珍椰, 波椰",
    "綠茶 大杯L 40 中杯M: 35 溫度必選: 正常冰, 少冰, 微冰, 去冰, 常溫, 溫, 熱 甜度必選:無糖, 9分甜,8分甜, 少糖, 6分甜, 半糖, 4分甜, 微糖,2分甜,1分甜,正常甜 加料可選: 珍珠, 波霸, 椰果, 混珠, 珍波椰, 珍椰, 波椰",
    "烏龍茶 大杯L 40 中杯M: 35 溫度必選: 正常冰, 少冰, 微冰, 去冰, 常溫, 溫, 熱 甜度必選:無糖, 9分甜,8分甜, 少糖, 6分甜, 半糖, 4分甜, 微糖,2分甜,1分甜,正常甜 加料可選: 珍珠, 波霸, 椰果, 混珠, 珍波椰, 珍椰, 波椰",
    "四季春 大杯L 40 中杯M: 35 溫度必選: 正常冰, 少冰, 微冰, 去冰, 常溫, 溫, 熱 甜度必選:無糖, 9分甜,8分甜, 少糖, 6分甜, 半糖, 4分甜, 微糖,2分甜,1分甜,正常甜 加料可選: 珍珠, 波霸, 椰果, 混珠, 珍波椰, 珍椰, 波椰",
]
def process_menu_items():
    processed_items = []
    for idx, item in enumerate(menu_items, 1):
        # 拆分菜單項目
        name = item.split()[0]  # 飲料名稱
        sizes = re.findall(r'(大杯L|中杯M).*?(\d+)', item)  # 尺寸和價格
        temps = re.findall(r'溫度必選: (.*?)甜度', item)[0].split(', ')  # 溫度選項
        sweets = re.findall(r'甜度必選:(.*?)加料', item)[0].split(', ')  # 甜度選項
        addons = re.findall(r'加料可選: (.*?)$', item)[0].split(', ')  # 加料選項

        # 建立結構化資料
        processed_item = {
            "id": idx,
            "name": name,
            "sizes": dict([(s.replace('杯L','').replace('杯',''), p) for s, p in sizes]),
            "temperatures": [t.strip() for t in temps],
            "sweetness": [s.strip() for s in sweets],
            "addons": [a.strip() for a in addons]
        }
        processed_items.append(processed_item)
    return processed_items
# ----------------------------
# 建立向量資料庫 (FAISS)
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(menu_items)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))



def get_relevant_menu(user_input: str, k: int = 2) -> List[Dict]:
    # 處理使用者輸入
    input_lower = user_input.lower()
    
    # 關鍵詞權重
    keywords = {
        "紅茶": 2.0, "綠茶": 2.0, "烏龍茶": 2.0, "四季春": 2.0,  # 飲品名稱加權
        "大": 1.5, "中": 1.5,  # 尺寸加權
    }
    
    # 建立加權查詢向量
    weighted_input = user_input
    for keyword, weight in keywords.items():
        if keyword in input_lower:
            # 重複關鍵詞來增加權重
            weighted_input += f" {keyword * int(weight)}"
    
    # 向量檢索
    query_vec = embed_model.encode([weighted_input])
    D, I = index.search(np.array(query_vec), k=k)
    
    retrieved_menu = []
    processed_items = process_menu_items()
    
    for distance, idx in zip(D[0], I[0]):
        # 計算相似度分數
        base_similarity = 1 / (1 + distance)
        
        # 額外的相關性檢查
        item = processed_items[idx]
        relevance_score = 0.0
        
        # 檢查飲料名稱匹配
        if item["name"] in user_input:
            relevance_score += 0.3
            
        # 檢查尺寸匹配
        for size in item["sizes"].keys():
            if size in user_input:
                relevance_score += 0.2
                
        # 檢查溫度匹配
        for temp in item["temperatures"]:
            if temp in user_input:
                relevance_score += 0.2
                
        # 檢查甜度匹配
        for sweet in item["sweetness"]:
            if sweet in user_input:
                relevance_score += 0.2
                
        # 檢查加料匹配
        for addon in item["addons"]:
            if addon in user_input:
                relevance_score += 0.1
        
        # 綜合分數
        final_similarity = (base_similarity + relevance_score) / 2
        
        # 只返回相關性足夠高的結果
        if final_similarity > 0.4:
            menu_info = {
                "id": item["id"],
                "menu": menu_items[idx],
                "similarity": final_similarity,
                "details": item
            }
            retrieved_menu.append(menu_info)
    
    # 根據最終相似度排序
    retrieved_menu.sort(key=lambda x: x["similarity"], reverse=True)
    return retrieved_menu

# ----------------------------
# 生成 JSON 函數 (RAG)
# ----------------------------


def generate_order_json(user_input, retrieved_menu, context_history=None):
    context_text = ""
    if context_history:
        # 限制歷史對話長度
        recent_history = context_history[-3:]  # 只保留最近3輪對話
        context_text = "\n".join([f"使用者: {c['user']}\n系統: {c['bot']}" for c in recent_history])

    # 將檢索到的菜單格式化，突出相似度高的選項
    menu_context = "檢索到的相關菜單:\n"
    most_relevant_item = None
    for item in retrieved_menu:
        if item["similarity"] > 0.6:  # 只使用相似度較高的結果
            details = item["details"]
            most_relevant_item = details
            menu_context += f"""- {details['name']}:
  • ID: {details['id']}
  • 可選尺寸: {', '.join(f'{size}({price}元)' for size, price in details['sizes'].items())}
  • 溫度選項: {', '.join(details['temperatures'])}
  • 甜度選項: {', '.join(details['sweetness'])}
  • 加料選項: {', '.join(details['addons'])}
  • 相關度: {item['similarity']:.2f}\n"""
    prompt = f"""
你是一個飲料店點餐系統。
使用者輸入：{user_input}

根據系統分析：
{f'• 最相關的飲品是 {most_relevant_item["name"]} (ID: {most_relevant_item["id"]})' if most_relevant_item else '• 無法確定最相關飲品'}

請根據以下格式將使用者需求轉成 JSON：

{{"items": [{{
    "id": 數字(1-4),
    "name": "飲料名稱",
    "count": 數量,
    "size": "中/大",
    "sweetness": "甜度",
    "temperature": "溫度",
    "addons": ["加料1", "加料2"]
}}]}}

注意:
- id: 紅茶=1, 綠茶=2, 烏龍茶=3, 四季春=4
- name 必須是: 紅茶, 綠茶, 烏龍茶, 四季春 其中之一
- size 必須是: 中, 大 其中之一
- sweetness 必須是: 無糖, 9分甜, 8分甜, 少糖, 6分甜, 半糖, 4分甜, 微糖, 2分甜, 1分甜, 正常甜 其中之一
- temperature 必須是: 正常冰, 少冰, 微冰, 去冰, 常溫, 溫, 熱 其中之一
- addons 可選項目必須是: 珍珠, 波霸, 椰果, 混珠, 珍波椰, 珍椰, 波椰 其中的零個或多個

請直接輸出 JSON，不要有其他文字。
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        # "format": {
        #     "type": "object",
        #     "properties": {
        #         "items": {
        #             "type": "array",
        #             "items": {
        #                 "type": "object",
        #                 "properties": {
        #                     "id": {"type": "integer", "enum": [1, 2, 3, 4]},
        #                     "name": {"type": "string", "enum": ["紅茶", "綠茶", "烏龍茶", "四季春"]},
        #                     "count": {"type": "integer", "minimum": 1},
        #                     "size": {"type": "string", "enum": ["中", "大"]},
        #                     "sweetness": {"type": "string", "enum": ["無糖", "9分甜", "8分甜", "少糖", "6分甜", "半糖", "4分甜", "微糖", "2分甜", "1分甜", "正常甜"]},
        #                     "temperature": {"type": "string", "enum": ["正常冰", "少冰", "微冰", "去冰", "常溫", "溫", "熱"]},
        #                     "addons": {
        #                         "type": "array",
        #                         "items": {"type": "string", "enum": ["珍珠", "波霸", "椰果", "混珠", "珍波椰", "珍椰", "波椰"]}
        #                     }
        #                 },
        #                 "required": ["id", "name", "count", "size", "sweetness", "temperature", "addons"]
        #             }
        #         }
        #     },
        #     "required": ["items"]
        # }
    }
    resp = requests.post(OLLAMA_API, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data


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

    try:
        retrieved_menu = get_relevant_menu(prompt)
        order_json_str = generate_order_json(
            prompt, retrieved_menu, context_history)

        print(f"\n測試訂單: {prompt}")
        print("模型回應:", order_json_str["response"])


        # 嘗試解析模型回應
        try:
            order = json.loads(order_json_str["response"])
            print("✅ 模型輸出成功解析成 JSON")
        except json.JSONDecodeError as e:
            print("❌ 模型輸出不是合法 JSON:", e)
            return False

        # 比對結構和內容
        if order == expected:
            print("✅ 完全匹配")
            context_history.append({"user": prompt, "bot": order_json_str["response"]})
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
