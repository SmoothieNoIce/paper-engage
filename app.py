from datetime import datetime
import os
import shutil
import json
import re
import requests
import argparse

from openai import OpenAI
from mitreattack.stix20 import MitreAttackData
from flask import Flask, request, jsonify
from sigma.rule import SigmaRule, SigmaDetections

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

from DDQN import DQN_agent
from utils import evaluate_policy, str2bool
from affect import predefined_actions

all_techniques = {}
label_encoder = None  

writer = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

client = None

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

agent: DQN_agent = None
attack_mapping_simple = None

def get_ttp_by_command(command):
    
    json_template = '''
    {
        "tactic": "",
        "techniques": [
            {
                "technique_id": "",
                "name": "",
                "procedure": ""
            }
        ]
    }
    '''
    prompt = f"""
    Analyze the following attack script and detect the TTPs (Tactics, Techniques, and Procedures) used based on the MITRE ATT&CK framework.

    Return **only** a response in the following strict JSON format:

    {json_template}

    Here is the attack script:

    {command}
    """
    response = client.chat.completions.create(model="gpt-4o-mini",  # 使用最新的模型
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are an expert in cybersecurity and MITRE ATT&CK."},
        {"role": "user", "content": prompt}
    ])

    # 解析返回的結果
    result = response.choices[0].message.content

    try:
        # 將 JSON 結果轉為 Python 字典格式
        result_dict = json.loads(result)
        return result_dict
    except json.JSONDecodeError as e:
        print("JSON decoding failed:", e)
        return None

def get_bert_embedding(command):
    inputs = tokenizer(command, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # 平均所有的 command embedding

def get_technique_as_int(technique_code):
    if technique_code in label_encoder.classes_:
        return label_encoder.transform([technique_code])[0]
    else:
        return -1  # 不在已知範圍內時返回 -1

def get_vector_by_command(command):
    bert_embedding = get_bert_embedding(command)
    return bert_embedding

def detect_powershell_content(content):
    # 檢查內容是否包含 PowerShell script 的關鍵字
    if "PowerShell" in content or "Get-" in content:
        return True
    else:
        return False

def preprocess_command(id, sid, command):
    is_script = detect_powershell_content(command)
    commandvec = get_vector_by_command(command)
    ttp = get_ttp_by_command(command)
    ttp_label_int = None
    
    if ttp == None or len(ttp['techniques']) == 0:
        ttp_label_int = get_technique_as_int('TNotFound')
    else:
        ttp_label_int = get_technique_as_int(ttp['techniques'][0]['technique_id'])

    return {
        'cmd': command,   
        'is_script': is_script,
        'commandvec': commandvec,
        'ttp_label_int': ttp_label_int,
        'ttp': ttp
    }

class ad_env:
    def __init__(self):
        self.isFirstCommand = True # 是否為第一個指令
        self.s_before = []
        self.a_before = []
        self.all_technique = []
        self.is_cleaning = False

        self.command_data_before = None
        self.command_before = ""
        self.technique_int_before = -1
        self.ttp_before = None
        self.total_steps = 0
        self.reward = 0
        self.total_steps = 0

        self.id_list = [] # 用於存儲每個指令的 ID(時間戳)
        pass

    def execute_action(self, state, sid):
        my_data = {'sid': sid, "action": 1}
        #r1 = requests.get(f"{app.config['engage_ad']}", data = my_data)
        #r2 = requests.post(f"{app.config['engage_network']}/post", data = my_data)
        pass
    
    def get_reward(self, command_data, ttp_before, a_before):
        current_reward = 0
        
        # Reward: MITRE Engage
        for action in predefined_actions:
            # 先檢查 action 與 predefined_actions 的對應
            if action['id'] == a_before:
                # 檢查每個 action 的 TTP
                engages_activity = action['activities']
                for ttp in ttp_before['techniques']:
                    if ttp['technique_id'] in list(attack_mapping_simple.keys()):
                        ttp_activity = attack_mapping_simple[ttp['technique_id']]
                        intersection = list(set(engages_activity) & set(ttp_activity))
                        if len(intersection) > 0:
                            current_reward += 10
                        else:
                            current_reward -= 5

        # Reward: MITRE ATT&CK
        for ttp in ttp_before['techniques']:
            if ttp['technique_id'] in self.all_technique:
                current_reward -= 20
            else:
                current_reward += 5

            self.all_technique.append(ttp['technique_id'])

        if "cd" in command_data['cmd'] or "Set-Location" in command_data['cmd']:
            current_reward += 5

        if "Invoke-Mimikatz" in command_data['cmd']:
            current_reward += 5

        #if "IEX" in command_data['cmd'] or "Invoke-Expression" in command_data['cmd']:
            #current_reward += 5
            # Meterpreter 有些普通的指令也會有 IEX，所以這裡不加分

        if "IWR" in command_data['cmd'] or "Invoke-WebRequest" in command_data['cmd']:
            current_reward += 5

        if "Get-ACL" in command_data['cmd']:
            current_reward += 5
        
        if "Get-Service" in command_data['cmd']:
            current_reward += 5
        
        technique_set = set(self.all_technique)
        writer.add_scalar('techniques', len(technique_set), global_step=adenv.total_steps)
            
        return current_reward

adenv: ad_env = None

# step for powershell command
@app.route('/api/ps', methods=['POST'])
def next_step():
    global adenv
    global writer
    global agent

    if request.is_json:
        data = request.get_json()

        if adenv.is_cleaning:
            '''清理環境'''
            print(f"Papar-Attack is cleaning")
            command = data['command']
            return jsonify({"message": "Papar-Attack is cleaning", "command": command, "result": 99 }), 200

        if data['status'] == 'success':
            '''讀取 command'''
            command = data['command'] # 指令內容
            id = data['id']           # 指令 ID
            sid = data['sid']         # 使用者 ID

            '''儲存序列'''
            adenv.id_list.append(id)

            '''分析目前 command 的 TTP，並儲存作為計算 Reward'''
            command_data = preprocess_command(id, sid, command)

            '''分析上個指令是否離開 Shell (die or win)'''
            '''TOdo: 改為分析上個指令是否離開 RDP (die or win)'''
            dw = 0 # 因為還沒離開 shell，所以為 win

            '''DQN，計算 action'''
            s_next: torch.Tensor = command_data['commandvec']
            if adenv.isFirstCommand:
                '''對當前的指令選擇 action'''
                a_next = agent.select_action(s_next, deterministic=False)
                adenv.execute_action(a_next, sid)
                adenv.isFirstCommand = False
            else:
                '''分析上一個指令 reward'''
                '''Todo: 比較上一個指令 technique 和上一個 action 的 activity 對應到的 technique 是否有重疊，'''
                '''還有目前環境的狀態，計算 reward'''

                r_before = adenv.get_reward(adenv.command_data_before, adenv.ttp_before, adenv.a_before)
                writer.add_scalar('action', adenv.a_before, global_step=adenv.total_steps)
                writer.add_scalar('reward', r_before, global_step=adenv.total_steps)

                '''計算上一個指令的 reward，然後進行學習'''
                agent.replay_buffer.add(adenv.s_before.detach().numpy(), adenv.a_before, r_before, s_next.detach().numpy(), dw)
                '''對當前的指令選擇 action'''
                a_next = agent.select_action(s_next, deterministic=False)
                adenv.execute_action(a_next, sid)

            '''Update AD Env，Next Step'''
            adenv.command_data_before = command_data
            adenv.technique_int_before = command_data['ttp_label_int']
            adenv.ttp_before = command_data['ttp']

            '''Rest API'''

            '''儲存 State 作為計算 reward'''
            adenv.s_before = s_next
            adenv.a_before = a_next

            '''Update Q Network'''
            if adenv.total_steps != 0 and adenv.total_steps % opt.update_every == 0:
                loss = agent.train()
                print('loss: {}'.format(loss))
                if opt.write:
                    writer.add_scalar('loss', loss, global_step=adenv.total_steps)

            '''Record & Log'''
            if adenv.total_steps % 50 == 0: 
                print('steps: {}'.format(int(adenv.total_steps)))
            
            '''save model'''
            if adenv.total_steps % opt.save_interval == 0:
                agent.save("DDQN",int(adenv.total_steps/10), adenv.total_steps)

            adenv.total_steps += 1
            
            '''Print Log'''
            try:
                # 檢查 command 並轉換非 ASCII 字符
                safe_command = command.encode('ascii', errors='replace').decode('ascii')
                print(f"message: PS command received successfully!, command:{safe_command}, result: {a_next}")
                for technique in command_data['ttp']['techniques']:
                    print(f"message: technique: {technique['technique_id']} - {technique['name']}")
            except Exception as e:
                print(f"Error occurred while printing: {e}")

            '''HTTP Response'''
            return jsonify({"message": "PS command received successfully!", "command": command, "result": a_next }), 200
        else:
            dw = 1
            s_next = get_vector_by_command("exit")
            r_before = adenv.get_reward(adenv.ttp_before, adenv.a_before)
            if len(adenv.s_before) != 0:
                agent.replay_buffer.add(adenv.s_before.detach().numpy(), adenv.a_before, r_before, s_next.detach().numpy(), dw)
            print(f"message: exit!")
            return jsonify({"message": "exit successfully!"}), 200

# Paper-Attack 環境清理
@app.route('/api/clean', methods=['POST'])
def clean():
    global adenv
    adenv.is_cleaning = True
    return jsonify({"message": "OK!" }), 200

# Paper-Attack 攻擊中
@app.route('/api/attack', methods=['POST'])
def attack():
    global adenv
    adenv.is_cleaning = False
    return jsonify({"message": "OK!" }), 200

if __name__ == '__main__':

    print("======!!!Paper Engage by:Flexolk!!!=====")

    print("======Reading Config=====")
    ## config
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    app.config.update(config)

    # OpenAI API
    client = OpenAI(api_key=app.config['openai_api_key'])

    print("======Reading Argument=====")
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--dvc', type=str, default='cuda:0', help='running device: cuda or cpu')
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--IsTrain', type=str2bool, default=True, help='Is Train or Not')

    parser.add_argument('--save_interval', type=int, default=int(50), help='Model saving interval, in steps.')
    parser.add_argument('--update_every', type=int, default=5, help='training frequency')
    parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
    parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
    parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')

    opt = parser.parse_args()
    opt.dvc = torch.device(opt.dvc)
    # env: real pc in active directory
    opt.state_dim = 768
    opt.action_dim = 8

    #Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    print(f"state_dim: {opt.state_dim}")
    print(f"action_dim: {opt.action_dim}")
    print(f"algo_name: {algo_name}")

    adenv: ad_env = ad_env()
    agent = DQN_agent(**vars(opt))
    adenv.is_cleaning = False

    ## Log 設定
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_S_'.format(algo_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
        print(f"writepath: {writepath}")


    print("======Reading MITRE ATT&CK=====")
    ## 整理 MITRE ATT&CK 
    mitre_attack_data = MitreAttackData("./json/enterprise-attack.json")
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)
    techniques_id = []
    all_techniques = {}
    for technique in techniques:
        attack_id = mitre_attack_data.get_attack_id(technique['id'])
        techniques_id.append(attack_id)
        all_techniques[attack_id] = technique['name']

    all_techniques['TNotFound'] = 'TNotFound'
    label_encoder = LabelEncoder()
    label_encoder.fit(techniques_id)

    ## config
    with open('json/attack_mapping_simple.json', 'r') as attack_mapping_simple_file:
        attack_mapping_simple = json.load(attack_mapping_simple_file)


    cudnn.benchmark = True

    print("======Start Engage=====")
    app.run(host=app.config['host'], port=app.config['port'])
