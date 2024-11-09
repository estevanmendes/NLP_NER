
import time 
from multiprocessing import Pool
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def openai_chat(prompt,text,model="gpt-4o-mini-2024-07-18")->str:
    api_key=os.getenv('OPENAI_API_KEY')
    client=OpenAI(api_key=api_key)
    messages=[
              {
                "role": "system",
                "content": [
                  {
                    "type": "text",
                    "text": prompt
                  }
                ]
              },
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": text
                  }
                ]
              }
            ]

    response=client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1002,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    content=response.choices[0].message.content
    return content



def add_response(id,gpt_response):
    print(f'{id} - time: {time.time()}')
    with open('topics.csv','a+') as f:
        f.write(str(id)+','+gpt_response+'\n')



def run_parallel_openai_chat(prompt,texts,start)->list:
    
    with Pool(10) as p:
        for index,item in enumerate(texts[start:]):
            p.apply_async(add_response,args=(index+start,call_gpt_sleeping(prompt,item)))
        p.close()
        p.join()
        
def call_gpt_sleeping(*args,**kwargs)->str:

    results=openai_chat(*args,**kwargs)
    time.sleep(0.01) ## Em razão da limitação de chamadas por minuto
    return results



if __name__=="__main__":
    
    import pandas as pd
    texts=pd.read_csv('plain_text.csv')['plain_text'].to_list() 

    prompt="""Sabendo que o texto a seguir é de origem médica realize as seguintes etapas:
            1- analise o texto, 
            2 - sintetize o texto em 3 tópicos, ou palavras chaves
            3- retorne um json da seguinte forma:{"topicos":["topic1","topic2","topic3"]}
            4- caso o texto seja ruidoso e não apresente informações claras e objetivas, retorne um json vazio {"topicos":["","",""]}"""    

    run_parallel_openai_chat(prompt,texts,336)

