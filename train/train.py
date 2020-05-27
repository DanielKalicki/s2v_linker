import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.sentence_linker import SentenceLinker
# from models.sentence_linker_discriminator import SentenceLinkerDiscriminator
from batchers.wiki_full_links_batch import WikiLinksBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

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
        # l2l_loss = -1 * torch.sqrt(torch.sum(torch.pow(output[b, 0] - output[b, 1], 2))).to('cpu')
        # l2l_loss = torch.nn.functional.threshold(l2l_loss, -1, -1)
        for o in range(output_size):
            for t in range(target_cnt_list[b]):
                if valid:
                    bl_loss = F.mse_loss(output[b, o].expand(1, config['s2v_dim']),
                                        target[b, t].expand(1, config['s2v_dim']))
                else:
                    bl_loss = F.smooth_l1_loss(output[b, o].expand(1, config['s2v_dim']),
                                        target[b, t].expand(1, config['s2v_dim']))
                loss_res[o, t] = bl_loss
        # loss += loss_res.min()
        # print(len(target))
        # print(loss_res)
        loss_out = torch.sort(torch.min(loss_res, dim=0)[0])[0][0:output_size]
        loss += torch.mean(loss_out)
        # if not valid:
        #     loss += l2l_loss*0.01
    if reduction == 'mean':
        loss /= len(target)
    # print(loss)
    return loss


def train(model, discriminator, device, train_loader,
          optimizer, optimizer_discriminator, epoch):
    model.train()
    # discriminator_accuracy = 0
    train_loss = 0
    start = time.time()
    pbar = tqdm(total=len(train_loader), ascii=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for _ in range(config['training']['batch_repeat']):
            optimizer.zero_grad()
            # model prediction
            outputs = []
            # prev_link = torch.zeros((list(data.shape)[0], config['s2v_dim']), dtype=torch.float).to(device)
            prev_link = data
            for _ in range(config['sentence_linker']['num_gen_links']):
                output = model(data, prev_link)
                outputs.append(output)
                prev_link = output
            output = torch.stack(outputs, dim=1)

            # discriminator prediction
            # discriminator_batch, discriminator_labels = dataset_train.get_discriminator_batch(data)
            # discriminator_batch = discriminator_batch.to(device)
            # discriminator_labels = discriminator_labels.to(device)
            # optimizer_discriminator.zero_grad()
            # discriminator_output = discriminator(discriminator_batch)
            # discriminator_loss = F.binary_cross_entropy_with_logits(discriminator_output, discriminator_labels)

            # model training
            loss = loss_calc(output, target, reduction='mean') # - discriminator_loss*1e-2
            if config['training']['l1_loss'] > 0.0:
                l1_regularization = 0
                for param in model.parameters():
                    l1_regularization += torch.sum(torch.abs(param))
                loss += config['training']['l1_loss']*l1_regularization
            train_loss += loss.detach()
            loss.backward(retain_graph=True)
            optimizer.step()

            # discriminator training
            # discriminator_loss.backward()
            # optimizer_discriminator.step()

            # calculate discriminator accuracy
            # probs = torch.softmax(discriminator_output, dim=1)
            # winners = probs.argmax(dim=1)
            # target = discriminator_labels.argmax(dim=1)
            # corrects = (winners == target)
            # discriminator_accuracy += corrects.sum().float() / float(target.size(0))
        pbar.update(1)
    pbar.close()
    end = time.time()
    print("")
    # print('Epoch {}: Train set: Discriminator accuracy: {:.4f}'.format(epoch, discriminator_accuracy/(batch_idx+1)))
    # if config['training']['log']:
        # writer.add_scalar('acc/train_discr', discriminator_accuracy/(batch_idx+1), epoch)
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss/(batch_idx+1)))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss/(batch_idx+1), epoch)
        writer.flush()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = []
            # prev_link = torch.zeros((list(data.shape)[0], config['s2v_dim']), dtype=torch.float).to(device)
            prev_link = data
            for _ in range(config['sentence_linker']['num_gen_links']):
                output = model(data, prev_link)
                outputs.append(output)
                prev_link = output
            output = torch.stack(outputs, dim=1) # TODO validation model generates only one link
            test_loss += loss_calc(output, target, reduction='sum', valid=True).detach()

    test_loss /= len(test_loader.dataset)
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.flush()

    if epoch == 10 or True:
        print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceLinker(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)

# discriminator
model_discriminator = None
optimizer_discriminator = None
# model_discriminator = SentenceLinkerDiscriminator(config).to(device)
# optimizer_discriminator = optim.Adam(model_discriminator.parameters(),
                                     # lr=config['training']['lr_discriminator'])
# print(model_discriminator)

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
for epoch in range(1, config['training']['epochs'] + 1):
    train(model, model_discriminator, device, data_loader_train,
          optimizer, optimizer_discriminator, epoch)
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
