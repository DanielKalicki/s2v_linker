import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.sentence_linker import SentenceLinker
# from models.sentence_linker_discriminator import SentenceLinkerDiscriminator
# from batchers.wiki_full_links_batch import WikiLinksBatch
from batchers.wiki_first_sent_links_batch import WikiLinksBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
import pickle

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]


def loss_calc(output, target, reduction='mean', valid=False):
    output_size = list(output.shape)[1]
    loss = 0
    target_cnt_list = []
    for b in target:
        target_cnt = 0
        for t in b:
            if torch.sum(torch.abs(t)) < 10000:
                target_cnt += 1
        target_cnt_list.append(target_cnt)
    for b in range(len(target)):
        loss_res = torch.ones((output_size, target_cnt_list[b]))
        for o in range(output_size):
            for t in range(target_cnt_list[b]):
                bl_loss = F.mse_loss(output[b, o].expand(1, config['s2v_dim']),
                                     target[b, t].expand(1, config['s2v_dim']))
                loss_res[o, t] = bl_loss
        loss_out = torch.sort(torch.min(loss_res, dim=0)[0])[0][0:output_size]
        loss += torch.mean(loss_out)
    if reduction == 'mean':
        loss /= len(target)
    # print(loss)
    return loss

def loss_fix_pos(output, target, target_mask, reduction="mean", valid=False):
    output = output.view(-1, config['s2v_dim'])
    target = target.view(-1, config['s2v_dim'])
    target_mask = target_mask.view(-1, config['s2v_dim'])
    s2v_dim = torch.tensor(config['s2v_dim'])
    loss = F.mse_loss(output*target_mask, target*target_mask, reduction='sum')/s2v_dim
    loss = loss/(torch.sum(target_mask)/s2v_dim)
    return loss

def test(model, device, test_loader, epoch, model_denoising=None):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target, target_mask) in enumerate(test_loader):
            data, target, target_mask = data.to(device), target.to(device), target_mask.to(device)
            if config['sentence_linker']['rnn_model']:
                outputs = []
                if config['sentence_linker']['state_vect']:
                    prev_state = torch.zeros(data.shape[0], config['sentence_linker']['prev_link_hdim']).to(device)
                else:
                    prev_state = data
                config['sentence_linker']['num_gen_links'] = 6
                for _ in range(config['sentence_linker']['num_gen_links']):
                    if config['sentence_linker']['state_vect']:
                        output, state = model(data, prev_state)
                    else:
                        output = model(data, prev_state)
                        state = output
                    outputs.append(output)
                    prev_state = state
                output = torch.stack(outputs, dim=1)
            else:
                output = model(data, None)
                output = output.view(-1, config['sentence_linker']['num_gen_links'], config['s2v_dim'])
            for i in range(config['sentence_linker']['num_gen_links']):
                dataset_test.check_accuracy(output[0, i].cpu().numpy())
                print("\t---------------")
            test_loss += loss_calc(output, target, reduction='sum', valid=True).detach()
    test_loss /= len(test_loader.dataset)
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceLinker(config)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)
checkpoint = torch.load('./train/save/'+config['name'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
del checkpoint

dataset_train = WikiLinksBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)

dataset_test = WikiLinksBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1,
    shuffle=False, num_workers=0)

test_loss = 1e6
for epoch in range(1, 2):
    current_test_loss = test(model, device, data_loader_test, epoch)
    print(current_test_loss)
    if current_test_loss < test_loss:
        test_loss = current_test_loss
