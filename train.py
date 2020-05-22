import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.sentence_linker import SentenceLinker
from models.sentence_linker_discriminator import SentenceLinkerDiscriminator
from batchers.wiki_links_batch import WikiLinksBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs


print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]


if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])


def loss_calc(output, target, reduction='mean'):
    loss = 0
    for b in range(len(target)):
        b_loss = 1e6
        for l in range(len(target[b])):
            bl_loss = F.mse_loss(output[b].expand(1, config['s2v_dim']),
                                 target[b, l].expand(1, config['s2v_dim']))
            if b_loss > bl_loss:
                b_loss = bl_loss  # min value
        loss += b_loss
    if reduction == 'mean':
        loss /= len(target)
    return loss

# def loss_calc(output, target, reduction='mean'):
#     output_size = 1
#     loss = 0
#     target_cnt_list = []
#     for b in target:
#         target_cnt = 0
#         for t in b:
#             if torch.sum(torch.abs(t)) < 10000:
#                 target_cnt += 1
#         target_cnt_list.append(target_cnt)
#     for b in range(len(target)):
#         loss_res = torch.ones((output_size, target_cnt_list[b]))
#         for o in range(output_size):
#             for t in range(target_cnt_list[b]):
#                 bl_loss = F.mse_loss(output[b].expand(1, config['s2v_dim']),
#                                      target[b, t].expand(1, config['s2v_dim']))
#                 loss_res[o, t] = bl_loss
#         # loss += loss_res.min()
#         loss_out = torch.sort(torch.min(loss_res, dim=0)[0])[0][0:output_size]
#         loss += torch.mean(loss_out)
#     if reduction == 'mean':
#         loss /= len(target)
#     return loss


def train(model, discriminator, device, train_loader,
          optimizer, optimizer_discriminator, epoch):
    model.train()
    discriminator_accuracy = 0
    train_loss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for i in range(config['training']['batch_repeat']):
            optimizer.zero_grad()
            # model prediction
            output = model(data)

            # discriminator prediction
            # discriminator_batch, discriminator_labels = dataset_train.get_discriminator_batch(data)
            # discriminator_batch = discriminator_batch.to(device)
            # discriminator_labels = discriminator_labels.to(device)
            # optimizer_discriminator.zero_grad()
            # discriminator_output = discriminator(discriminator_batch)
            # discriminator_loss = F.binary_cross_entropy_with_logits(discriminator_output,
                                                                    # discriminator_labels)

            # model training
            loss = loss_calc(output, target, reduction='mean') # - \
    #               discriminator_loss*1e-2
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
            output = model(data)
            test_loss += loss_calc(output, target, reduction='sum').detach()
            # test_loss += F.mse_loss(output, target, reduction='sum').item()
            # cosine similarity?

    test_loss /= len(test_loader.dataset)
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.flush()

    if epoch == 10 or True:
        print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))


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

# scheduler = lr_scheduler(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, config['training']['epochs'] + 1):
    train(model, model_discriminator, device, data_loader_train,
          optimizer, optimizer_discriminator, epoch)
    test(model, device, data_loader_test, epoch)
    # scheduler.step()

if config['training']['log']:
    writer.close()

# torch.save(model.state_dict(), "mnist_cnn.pt")
