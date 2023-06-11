# https://github.com/Ruu3f/freeGPT/tree/main
# pip install freeGPT
# from freeGPT import alpaca, gpt3, gpt4
# pip install gpt4free
import sys
sys.path.append("D:\\WorkSpace\\repo\\gpt4free")
import gpt4free

def ask_alpaca(prompt):
    resp = alpaca.Completion.create(prompt=prompt)
    return resp

def ask_gpt3(prompt):
    resp = gpt3.Completion.create(prompt=prompt)
    return resp['text']

def ask_gpt4(prompt):
    token = gpt4.Account.create(logging=True)
    resp = gpt4.Completion.create(prompt=prompt, token=token)
    return resp.text

def ask_gpt4free(prompt):
    # You
    def You(model, chat):
        from gpt4free import you
        response = you.Completion.create(
            prompt=prompt,
            detailed=True,
            include_links=True, chat=chat)
        
        chat.append({"question": prompt, "answer": response.text})
        return response.text
    # Theb
    def Theb(model, chat):
        pass
    # ForeFront
    def ForeFront(model, chat):
        from gpt4free import forefront
        # create an account
        account_data = forefront.Account.create(logging=False)

        # get a response
        result = ""
        for response in forefront.StreamingCompletion.create(
            account_data=account_data,
            prompt='hello world',
            model='gpt-4'
        ):
            result = result + response.choices[0].text + " "
        return result
    # Quora
    def Quora(model, chat):
        from gpt4free import quora
        token = "_I8F4IkOawOpj-lnilgfmg%3D%3D"
        response = quora.StreamingCompletion.create(custom_model=model, prompt=prompt, token=token)
        print(response)
        for i in response:
            print(i)
        return response.completion.choices[0].text
    
    gpt4free_dict = {}
    gpt4free_dict['Quora'] = Quora
    gpt4free_dict['ForeFront'] = ForeFront
    gpt4free_dict['You'] = You
    gpt4free_dict['Theb'] = Theb
    
    # gpt4free_func = [('Quora', 'GPT-4'), ('ForeFront', None), ('You', None), ('Theb', None)]
    gpt4free_func = [('Quora', 'gpt-4')]
    # for model in ['Sage', 'Claude+', 'Claude-instant', 'ChatGPT', 'Dragonfly', 'NeevaAI']:
    #     gpt4free.append(('Quora', model))
    
    chat = []
    for name, model in gpt4free_func:
        try:
            resp = gpt4free_dict[name](model, chat)
            print(f"🤖 {name, model} > {resp}")
        except Exception as e:
            print(f"🤖 {name, model} > I failed becasue {e}")
    

gpts = {}
# gpts["alpaca"] = ask_alpaca
# gpts["gpt3"] = ask_gpt3
# gpts["gpt4"] = ask_gpt4
gpts["gpt4free"] = ask_gpt4free

if __name__ == "__main__":
    while True:
        prompt = input("👦 > ")
        for ai in gpts:
            try:
                resp = gpts[ai](prompt)
                print(f"🤖 {ai} > {resp}")
            except Exception as e:
                print(f"🤖 {ai} > I failed becasue {e}")
        