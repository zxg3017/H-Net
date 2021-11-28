'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import models.loss.Loss_all as Loss
import models.metrics.Miou_bak as Miou
from data.create_dataset import CreateDataset
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
from models.my_H_Net_model.H_Net import H_Net_resnet34_double_v137
import utils as utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#### train dataset
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/images/', help='imgs train data path.')
# parser.add_argument('--imgs_coarse_mask_path', '-it', type=str,
#                     default='/mnt/ai2019/zxg_FZU/dataset/skin_lesion_data/seg/train/Annotation/', help='coarse_mask train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/masks/', help='labels train data path.')
#### val dataset
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/images/', help='imgs val data path.')
# parser.add_argument('--imgs_val_path', '-iv', type=str,
#                     default='/mnt/ai2019/zxg_FZU/dataset/skin_lesion_data/seg/val/Annotation/', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/masks/', help='labels val data path.')
parser.add_argument('--resize', default=256, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=200, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/CE_Net_101_28/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/H_Net/wbc/', help='checkpoint path')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--devicenum', default='2', type=str, help='use devicenum')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
torch.backends.cudnn.enabled =True
tb_path = args.tb_path
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
device = args.device # 是否使用cuda
best_miou = 0  # best test accuracy
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
times = args.times  # 验证次数
checkpoint_path = args.checkpoint + 'wbc_H_Net_resnet34_double_v137_hy46.pth'
checkpoint_path_latest = args.checkpoint + 'wbc_H_Net_resnet34_double_v137_hy46_latest.pth'
# checkpoint_path = '/media/user/DA18EBFA09C1B27D/codess/nodulesegmentation/checkpoint/Unet/ckpt.pth'
# Data
print('==> Preparing data..')


trainset = CreateDataset(img_paths=args.imgs_train_path, label_paths=args.labels_train_path,
                         resize=args.resize, phase='train', aug=True)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

valset = CreateDataset(img_paths=args.imgs_val_path, label_paths=args.labels_val_path,
                         resize=args.resize, phase='val', aug=False)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Model
print('==> Building model..')
# net = new.N_unet(3, 2)
net = H_Net_resnet34_double_v137(3,2)
model_name = 'wbc_H_Net_resnet34_double_v137_hy46'
print(model_name)
# net = resnet_test.resnet_uunet(3,2)
# net = resnet_test.resnet_unet_attention(3,2)

print("param size = %fMB", utils.count_parameters_in_MB(net))

EPS = 1e-12
net = net.to(device)
# print(args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path_latest)
    net.load_state_dict(checkpoint['net'])
    best_miou = checkpoint['miou']
    start_epoch = checkpoint['epoch']

criterion = Loss.CrossEntropyLoss2D().to(device)
# criterion = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()

ce_Loss = Loss.CrossEntropyLoss2D().to(device)
dice_Loss = Loss.myDiceLoss(2).to(device)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-8)

def train_val():
    with SummaryWriter(tb_path) as write:
        train_step = 0
        for epoch in range(start_epoch, args.end_epoch):
            with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
                net.train()
                train_loss = 0
                train_miou = 0
                train_pa = 0

                for batch_idx, (inputs, targets, img_path) in enumerate(trainloader):
                    t.set_description("Train(Epoch{}/{})".format(epoch, args.end_epoch))
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    out1,out2= net(inputs)

                    ce_loss = ce_Loss(torch.log(softmax_2d(out1) + EPS), targets)
                    dice_loss = dice_Loss(torch.log(softmax_2d(out2) + EPS), targets)
                    loss = 0.4 * ce_loss + 0.6 * dice_loss

                    loss.backward()
                    optimizer.step()
                    out = (out1 + out2) / 2
                    out = torch.log(softmax_2d(out) + EPS)

                    train_loss += loss.item()
                    predicted = out.argmax(1)
                    # predicted = outputs.argmax(1)
                    # train_miou += Miou.calculate_miou(predicted, targets, 2).item()
                    train_miou += Miou.Pa(predicted,targets).item()

                    write.add_scalar('Train_loss', train_loss / (batch_idx + 1), global_step=train_step)
                    # write.add_scalar('Train_Miou', 100. * (train_miou / (batch_idx + 1)), global_step=train_step)
                    write.add_scalar('train_pa', 100. * (train_miou / (batch_idx + 1)), global_step=train_step)
                    train_step += 1
                    t.set_postfix(loss='{:.3f}'.format(train_loss / (batch_idx + 1)),
                                  train_pa='{:.2f}%'.format(100. * (train_miou / (batch_idx + 1))))
                    t.update(1)
                f = open('./loss_txt/'+model_name+'.txt', "a")
                loss_tar = "%.4f" % (train_loss/(batch_idx+1))
                pa_here = "%.4f" %(train_miou / (batch_idx + 1))
                f.write('epoch' + str(epoch+1) +'__all_loss'+str(loss_tar)+ '_train_pa'+str(pa_here))
                f.close()
                CosineLR.step()

            if epoch % times == 0:#多少个epoch验证一次
                global best_miou
                net.eval()
                test_loss = 0
                val_miou = 0


                with torch.no_grad():
                    with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
                        for batch_idx, (inputs, targets,img_path) in enumerate(valloader):
                            t.set_description("Val(Epoch {}/{})".format(epoch, args.end_epoch))
                            inputs, targets = inputs.to(device), targets.to(device)

                            out1, out2= net(inputs)

                            out = (out1 + out2) / 2

                            out = torch.log(softmax_2d(out) + EPS)

                            test_loss += loss.item()

                            predicted = out.argmax(1)

                            # val_miou += Miou.calculate_miou(predicted, targets, 2).item()
                            val_miou += Miou.Pa(predicted, targets).item()
                            t.set_postfix(loss='{:.3f}'.format(test_loss / (batch_idx + 1)),
                                          val_pa='{:.2f}%'.format(100. * (val_miou / (batch_idx + 1))))
                            t.update(1)
                        write.add_scalar('Val_loss', test_loss / (batch_idx + 1), global_step=train_step)
                        write.add_scalar('Val_pa', 100. * (val_miou / batch_idx + 1), global_step=train_step)
                        # Save checkpoint.
                        pa_here = "%.4f" %(val_miou / (batch_idx + 1))
                        f = open('./loss_txt/'+model_name+'.txt', "a")
                        f.write('epoch' + str(epoch+1) +'_val_pa'+str(pa_here)+'\n')
                        f.close()

                    if val_miou > best_miou:
                        print('Saving..',str(pa_here))
                        state = {
                            'net': net.state_dict(),
                            'miou': val_miou,
                            'epoch': epoch,
                        }
                        if not os.path.isdir(args.checkpoint):
                            os.mkdir(args.checkpoint)
                        torch.save(state, checkpoint_path)
                        best_miou = val_miou

                    state = {
                        'net': net.state_dict(),
                        'miou': val_miou,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoint_path_latest)
                    # print('best_miou',best_miou)



if __name__ == '__main__':
    train_val()
