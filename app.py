import os
from flask import Flask, request, jsonify
import json
import re
from DDQN import Duel_Q_Net, DQN_agent
import argparse
from utils import evaluate_policy, str2bool
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.backends.cudnn as cudnn
import numpy as np
from transformers import BertTokenizer, BertModel
from sigma.rule import SigmaRule, SigmaDetections
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

agent: DQN_agent = None
total_steps = 0

def getVectorByCommand(command):
    bert_embedding = get_bert_embedding(command)
    return bert_embedding

def getTechniqueByCommand(command):
    return None

def getReward(command):
    # 如果攻擊者的指令為惡意指令，reward = 1
    return 1

def get_bert_embedding(command):
    inputs = tokenizer(command, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # 平均所有的詞嵌入


s_before = []
a_before = []
first = True
id_list = []

# Route to handle Wazuh-detected technique
@app.route('/api/ps', methods=['POST'])
def receive_command():
    global agent
    global s_before
    global a_before
    global  first
    global id_list

    if request.is_json:
        data = request.get_json()
        if data['status'] == 'success':
            # 分析 command
            command = data['command']
            id = data['id']
            commandvec = getVectorByCommand(command)
            #technique = getTechniqueByCommand(command)

            # 分析 dw
            dw = 0
            id_list.append(id)

            # 分析 reward
            r_before = getReward(command)

            s_next: torch.Tensor = commandvec
            #s_next = np.array([technique])
            #s_next = np.concatenate((s_next, commandvec), axis=None)
            if first:
                a_next = agent.select_action(s_next, deterministic=False)
                first = False
            else:
                agent.replay_buffer.add(s_before.detach().numpy(), a_before, r_before, s_next.detach().numpy(), dw)
                a_next = agent.select_action(s_next, deterministic=False)
            s_before = s_next
            a_before = a_next

            #sendHttpRequest("https://127.0.0.1:8080/api/ad?action=1")

            '''Update'''
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                for j in range(opt.update_every): agent.train()
                
            '''Noise decay & Record & Log'''

            '''save model'''
            my_data = {'sid': data['sid'], "action": a_next}
            r1 = requests.post(f"{app.config['engage_ad']}/post", data = my_data)
            r2 = requests.post(f"{app.config['engage_network']}/post", data = my_data)

            
            print(f"message: PS command received successfully!, command:{command}")
            return jsonify({"message": "PS command received successfully!", "command": command, "result": a_next }), 200
        else:
            dw = 1
            s_next = getVectorByCommand("exit")
            r_before = getReward("exit")
            agent.replay_buffer.add(s_before.detach().numpy(), a_before, r_before, s_next.detach().numpy(), dw)
            print(f"message: exit!")
            return jsonify({"message": "exit successfully!"}), 200

if __name__ == '__main__':

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    app.config.update(config)

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--dvc', type=str, default='cuda:0', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
    parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
    parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
    parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
    parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')

    test = getVectorByCommand("command")
    opt = parser.parse_args()
    opt.dvc = torch.device(opt.dvc)
    opt.state_dim = 768
    opt.action_dim = 16

    agent = DQN_agent(**vars(opt))

    cudnn.benchmark = True

    app.run(host='0.0.0.0', port=8080)

    """ if args.resume:
        testnetDQN = torch.load(args.netDQN)
    else:
        wandb.init(
            project = 'Engagement',
            config = {
                    "batch_size":args.batch_size, 
                    "epoch": 60, 
                    "embedding": "nn.Linear", 
                    "Type": "DQL",
                    "Block_size": 'bigger',
                    "Resume": False
            },
            name = "Test1"
        )
        train_dcgan.train(train_dataloader, test_dataloader)
        wandb.finish() """