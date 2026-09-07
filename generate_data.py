import json

# ตัวอย่างชุดข้อมูลบทสนทนาพื้นฐานที่เราต้องการสอน SmolLM
conversations = [
    # คำทักทายทั่วไป
    ("hi", "Hello! How can I help you today?"),
    ("hello", "Hi there! What can I do for you?"),
    ("สวัสดี", "สวัสดีครับ มีอะไรให้ผมช่วยเหลือไหมครับ?"),
    ("สวัสดีครับ", "สวัสดีครับ ยินดีที่ได้คุยกันครับ มีอะไรสอบถามได้เลยนะ"),
    
    # คำถามสั้นๆ ประจำวัน
    ("คุณคือใคร", "ผมคือ SmolLM AI อัจฉริยะขนาดเล็กที่รันอยู่บนเครื่องของคุณครับ"),
    ("Who are you?", "I am a local AI assistant running on your device!"),
    ("ทำอะไรได้บ้าง", "ผมสามารถตอบคำถามสั้นๆ ช่วยเขียนโค้ด หรือพูดคุยโต้ตอบทั่วไปได้ครับ"),
    
    # ภาษาไทยพื้นฐาน
    ("สบายดีไหม", "สบายดีครับ! พร้อมช่วยงานคุณเสมอครับ"),
    ("ขอบคุณนะ", "ยินดีมากๆ ครับ! มีอะไรให้ช่วยอีกบอกได้เลยนะ")
]

output_file = "smollm_thai_instruct.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for user_msg, assistant_msg in conversations:
        # จัดฟอร์แมตให้อยู่ในรูปแบบ ChatML
        formatted_text = (
            f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
        # เขียนลงไฟล์ JSONL
        json_line = json.dumps({"text": formatted_text}, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"สร้างไฟล์ {output_file} เรียบร้อยแล้ว!")

