import requests
import json
from datetime import datetime

web_urls = ["https://gpt4.gravityengine.cc/"]

def post_message(url, data):
    data = json.dumps(data, separators=(',', ':'))
    headers = {"Authority": "gpt4.gravityengine.cc", 
               "Method": "POST",
               "Path": "/api/openai/v1/chat/completions",
               "Scheme": "https", "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br", 
               "Accept-Language": "zh-CN,zh;q=0.9",
               "Content-Length": str(len(data)), 
               "Content-Type": "application/json", "Cookie": "FCNEC=%5B%5B%22AKsRol8uQRyGp-G2DVBqsmiaz1SuTrVnKK9UF-mC5UOoJBq8a85JR8K3QlWnAfctBnNsuVl73cW2-8WVneWKKy7HA1csPXO3eF1rX0gb-2hyuj5-pqYqj22xckkyAmqAymWmPUH0MHQLKPRtqXO9w0s9DZ1G3VRtug%3D%3D%22%5D%2Cnull%2C%5B%5D%5D; _ga_9SW3L5WRD8=GS1.1.1686361812.1.0.1686361812.0.0.0; _ga=GA1.1.1266994730.1686361813", 
               "Origin": "https://gpt4.gravityengine.cc",
               "Referer": "https://gpt4.gravityengine.cc/", 
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36", "X-Requested-With": "XMLHttpRequest"}
    print(headers)
    response = requests.post(url, data = data, headers=headers)
    print(response)
    # print(response.text)
    return response.text

def parse_rsp(data):
    data = data.split("\n\n")
    # print(data)
    rsp = ""
    for i in data:
        # print(i)
        if "[DONE]" in i:
            return rsp
        j = json.loads(i[6:])
        if "choices" in j:
            if "delta" in j["choices"][0]:
                if "content" in j["choices"][0]["delta"]:
                    rsp = rsp + j["choices"][0]["delta"]["content"] # + " "
    # print(rsp)
    return rsp
    

def gpt3_5(url, prompt):
    url = url + "api/openai/v1/chat/completions"
    payload = {}
    payload["model"] = "gpt-3.5-turbo"
    payload["stream"] = True
    payload["temperature"] = 0.5
    payload["messages"] = prompt
    rsp = post_message(url, payload)
    # print(rsp)
    return parse_rsp(rsp)
    # old_length = -1
    # rsp = None
    # while True:
    #     rsp = post_message(url, payload)
        # length = 0
        # for i in rsp["messages"]:
        #     if i["role"] == "assistant":
        #         length = len(i["content"])
        # if old_length > 0 and old_length == length:
        #     return rsp
        # else:
        #     old_length = length

def system_init_role():
    system_role = "IMPORTANT: You are a virtual assistant powered by the gpt-3.5-turbo model, now time is " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    msg = {}
    msg["role"] = "system"
    msg["content"] = system_role
    return [msg]

def add_user_prompt(chat):
    prompt = input("👦 > ")
    msg = {}
    msg["role"] = "user"
    msg["content"] = prompt
    chat.append(msg)
    return chat

def test_parse_rsp():
    with open("core/gpt/data.txt") as f:
        data = f.readlines()
        s = ""
        for i in data:
            s = s + i
        parse_rsp(s)

if __name__ == "__main__":
    # test_parse_rsp()
    # exit()
    while True:
        chat = system_init_role()
        chat = add_user_prompt(chat)
        try:
            resp = gpt3_5(web_urls[0], chat)
            print(f"🤖 > {resp}")
        except Exception as e:
            print(f"🤖 > I failed becasue {e}")
