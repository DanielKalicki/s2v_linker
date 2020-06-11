import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.sentence_linker import SentenceLinker
from models.sentence_linker_discriminator import SentenceLinkerDiscriminator
# from batchers.wiki_full_links_batch import WikiLinksBatch
from batchers.wiki_first_sent_links_batch import WikiLinksBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
import numpy as np

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]
load_model = 41


if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])


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

def loss_fix_pos(output, target, target_mask, valid=False):
    output = output.view(-1, config['s2v_dim'])
    target = target.view(-1, config['s2v_dim'])
    target_mask = target_mask.view(-1, config['s2v_dim'])
    s2v_dim = torch.tensor(config['s2v_dim'])
    loss = F.mse_loss(output*target_mask, target*target_mask, reduction='sum')/s2v_dim
    loss = loss/(torch.sum(target_mask)/s2v_dim)
    return loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    for batch_idx, (data, target, target_mask) in enumerate(train_loader):
        data, target, target_mask = data.to(device), target.to(device), target_mask.to(device)
        for _ in range(config['training']['batch_repeat']):
            optimizer.zero_grad()
            # model prediction
            if config['sentence_linker']['rnn_model']:
                outputs = []
                if config['sentence_linker']['state_vect']:
                    prev_state = torch.zeros(data.shape[0], config['sentence_linker']['prev_link_hdim']).to(device)
                else:
                    prev_state = data
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

            # model training
            loss = loss_calc(output, target, reduction='mean')
            # loss = loss_fix_pos(output, target, target_mask) # - discriminator_loss*1e-1
            if config['training']['l1_loss'] > 0.0:
                l1_regularization = 0
                for param in model.parameters():
                    l1_regularization += torch.sum(torch.abs(param))
                loss += config['training']['l1_loss']*l1_regularization
            train_loss += loss.detach()
            loss.backward(retain_graph=True)
            optimizer.step()

        pbar.update(1)
    pbar.close()
    end = time.time()
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss/(batch_idx+1)))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss/(batch_idx+1), epoch)
        writer.flush()


def test(model, device, test_loader, epoch):
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
            test_loss += loss_calc(output, target, reduction='sum', valid=True).detach()

    test_loss /= len(test_loader.dataset)
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceLinker(config)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)
start_epoch = 1
if load_model:
    checkpoint = torch.load('./train/save/'+configs[load_model]['name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch += checkpoint['epoch']
model.to(device)
if load_model:
    del checkpoint

dataset_train = WikiLinksBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)

dataset_test = WikiLinksBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=4)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + 1):
    train(model, device, data_loader_train, optimizer, epoch)
    current_test_loss = test(model, device, data_loader_test, epoch)
    if current_test_loss < test_loss:
        test_loss = current_test_loss
        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
            }, './train/save/'+config['name'])
    # scheduler.step()

if config['training']['log']:
    writer.close()
